from typing import Tuple
from collections import deque

import torch

_ACTIVATION_QUEUE = None

class ActivationQueue:
    """
    A queue of stack to record activation refs in an iteration
    
    Each microbatch's activations (maybe multiple layers) are organized as a stack
    Each iteration's activations (maybe multiple microbatches) are organized as a queue
    """
    def __init__(self, pipeline_size, pipeline_rank, num_microbatches,
                 num_layers) -> None:
        self.pp_rank = pipeline_rank
        self.pp_size = pipeline_size

        num_warmup_microbatches = (pipeline_size - pipeline_rank -1)
        num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches

        print(f"> {pipeline_rank}/{pipeline_size}: init activation queue at "
              f"{pipeline_rank}/{pipeline_size}, "
              f"num micro batches: {num_microbatches}, num layers: {num_layers}, "
              f"num warmup: {num_warmup_microbatches}, "
              f"num steady: {num_microbatches_remaining}")
        
        self.num_warmup_microbatches = num_warmup_microbatches
        self.num_microbatches_remaining = num_microbatches_remaining
        self.num_microbatches = num_microbatches
        self.num_stacks = min(num_warmup_microbatches + 1, num_microbatches)
        self.num_layers = num_layers

        self.queue = deque(maxlen=self.num_stacks)

        self.micro_batch_idx = 0
        self.layer_idx = 0

        self.free_list = deque(maxlen=self.num_stacks*num_layers-1)
        self.act_buf_map = {}
        self.offload_stream = torch.cuda.Stream()
        self.upload_stream = torch.cuda.Stream()

    def empty(self):
        return len(self.queue) == 0

    def prev_stack_full(self):
        return len(self.queue[-1]) == self.num_layers

    def push(self, activations: Tuple[torch.Tensor]):
        if self.empty() or self.prev_stack_full():
            self.queue.append(deque(maxlen=self.num_layers))

        self.queue[-1].append(activations)
        # print(f"> {self.pp_rank}/{self.pp_size} "
        #       f"PUSH: {(self.micro_batch_idx, self.layer_idx)} "
        #       f"at {len(self.queue), len(self.queue[-1])}", flush=True)

        self.layer_idx += 1
        if self.layer_idx == self.num_layers:
            self.micro_batch_idx += 1
            self.layer_idx = 0

    def upload_front_top(self):
        if self.num_layers == 1:
            return len(self.queue) == self.num_stacks - 1
        else:
            return self.num_stacks > 1 and \
                len(self.queue) == self.num_stacks and \
                len(self.queue[-1]) == self.num_layers - 1

    def get_free_buffer(self, tensors: Tuple[torch.Tensor]):
        if len(self.free_list) > 0:
            return self.free_list.popleft()
        buffer = []
        for each in tensors:
            buffer.append(torch.empty_like(each, device='cpu', pin_memory=True).untyped_storage())
        return buffer

    def _offload(self):
        back_top = self.queue[-1][-1]
        back_top_buffer = self.get_free_buffer(back_top)
        self.act_buf_map[back_top] = back_top_buffer
        # print(f"offload: {back_top[0][0].item()}, free list: {len(self.free_list)}")
        self.offload_stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(self.offload_stream):
            for act, buf in zip(back_top, back_top_buffer):
                buf.copy_(act.untyped_storage(), non_blocking=True)
                act.untyped_storage().resize_(0)

    def _upload(self):
        front_top = self.queue[0][-1]
        front_top_buf = self.act_buf_map.pop(front_top)
        self.free_list.append(front_top_buf)
        self.upload_stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(self.upload_stream):
            for act, buf in zip(front_top, front_top_buf):
                act.untyped_storage().resize_(act.nbytes)
                act.untyped_storage().copy_(buf, non_blocking=True)
        # print(f"upload: {front_top[0][0].item()}, free list: {len(self.free_list)}")

    def forward(self):
        if self.empty():
            # print(f"> {self.pp_rank}/{self.pp_size} "
            #       f"FORWARD: skip: {(self.micro_batch_idx, self.layer_idx)} ", 
            #       flush=True)
            return
        # print(f"> {self.pp_rank}/{self.pp_size} "
        #       f"FORWARD: offload: {len(self.queue), len(self.queue[-1])}", 
        #       flush=True)

        # `forward` is triggered before `push`
        # do not need to check if back is in the previous microbatch
        # offload back top
        self._offload()

        # upload front top for backward
        if self.upload_front_top():
            # print(f"> {self.pp_rank}/{self.pp_size} "
            #       f"FORWARD: upload: {len(self.queue)}, {len(self.queue[0])}", 
            #       flush=True)
            self._upload()

    def skip_upload_front_top(self):
        # print(f"{len(self.queue)}, {len(self.queue[0]) if len(self.queue) > 0 else None}, {self.micro_batch_idx}")
        if len(self.queue) == 0:
            return True
        
        if len(self.queue[0]) == self.num_layers:
            return (self.micro_batch_idx < self.num_microbatches or len(self.queue) == 1)

        return False

    def pop(self):
        # print(f"> {self.pp_rank}/{self.pp_size} "
        #       f"BACKWARD: pop: {len(self.queue)}, {len(self.queue[0])}",
        #       flush=True)

        # pop the top layer's activations
        self.queue[0].pop()
        # pop the current microbatch's activations if it popped the first layer
        if len(self.queue[0]) == 0:
            self.queue.popleft()

    def backward(self):
        # upload the new top layer for backward
        if self.skip_upload_front_top():
            return
        # print(f"> {self.pp_rank}/{self.pp_size} "
        #       f"BACKWARD: upload: {len(self.queue[0])}", flush=True)
        self._upload()
    
    def reset(self, sanity_check=False):
        if sanity_check:
            assert self.micro_batch_idx == self.num_microbatches
            assert self.layer_idx == 0
            assert len(self.queue) == 0
            assert len(self.free_list) == (self.num_stacks*self.num_layers-1)
        self.layer_idx = 0
        self.micro_batch_idx = 0


def set_activation_queue(pp_size, pp_rank, num_microbatches, num_layers):
    global _ACTIVATION_QUEUE
    _ACTIVATION_QUEUE = ActivationQueue(pp_size, pp_rank, num_microbatches, num_layers)


def get_activation_queue():
    assert _ACTIVATION_QUEUE is not None, 'activation queue is not initialized'
    return _ACTIVATION_QUEUE
