from functools import reduce
from operator import mul

import torch
import torch.distributed as dist

from megatron.core import parallel_state as mpu


class PipelineContext:
    """
    design for attention pipeline to handle two functions:
    1. the communication of activations and the corresponding gradients
    2. the offloading and uploading of activations
    """
    def __init__(self, args) -> None:
        assert args.attention_pipeline

        self.rank = mpu.get_pipeline_model_parallel_rank()
        self.wsize = mpu.get_pipeline_model_parallel_world_size()
        self.num_loop = mpu.get_virtual_pipeline_model_parallel_world_size()
        self.num_fold = args.num_fold
        self.verbose = False
        self.offload_verbose = False
        self.deallocate_pipeline_outputs = False #True

        self.input_tensors = [[[] for _ in range(self.wsize)] for _ in range(self.num_fold)]
        self.output_tensors = [[[] for _ in range(self.wsize)] for _ in range(self.num_fold)]
        self.reqs = [[] for _ in range(self.num_fold)]

        self.grad_buf = [None for _ in range(self.num_fold)]
        self.labels = []

        self.offload_activation = args.offload_activation and args.checkpoint_without_attn and args.transfer_weight
        if self.offload_activation:
            self.param_batch_idx = 0
            self.param_act_queue = [[] for _ in range(self.num_fold)]
            self.attn_batch_idx = self.num_fold * (self.wsize - self.rank)
            self.attn_act_queue = [[[] for _ in range(self.wsize-self.rank)] for _ in range(self.num_fold)]
            self.act_buf_map = {}
            self.free_list = {}
            self.offload_stream = torch.cuda.Stream()

    # functions in p2p_communication.py are no longer applicable
    def _communicate(self, config, recv_shapes=None, src_ranks=None, send_tensors=None, tgt_ranks=None, is_bwd=False, fold=None):
        ops = []
        ret = []
        device=torch.cuda.current_device()
        group = mpu.get_pipeline_model_parallel_group()
        if send_tensors is not None:
            assert len(send_tensors) == len(tgt_ranks)
            for tensor, tgt in zip(send_tensors, tgt_ranks):
                grp_rank = dist.get_global_rank(group, tgt)
                ops.append(dist.P2POp(dist.isend, tensor, grp_rank, group))
        
        if recv_shapes is not None:
            assert len(recv_shapes) == len(src_ranks)
            for shape, src in zip(recv_shapes, src_ranks):
                buf = torch.empty(
                    shape, requires_grad=True, 
                       dtype=config.pipeline_dtype, device=device)
                grp_rank = dist.get_global_rank(group, src)
                ops.append(dist.P2POp(dist.irecv, buf, grp_rank, group))
                ret.append(buf)
        reqs = dist.batch_isend_irecv(ops)
        return ret, reqs

    def reset(self,):
        self.input_tensors = [[[] for _ in range(self.wsize)] for _ in range(self.num_fold)]
        self.output_tensors = [[[] for _ in range(self.wsize)] for _ in range(self.num_fold)]
        self.reqs = [[] for _ in range(self.num_fold)]

        self.grad_buf = [None for _ in range(self.num_fold)]
        if self.offload_activation:
            self.param_batch_idx = 0
            self.param_act_queue = [[] for _ in range(self.num_fold)]
            self.attn_batch_idx = self.num_fold * (self.wsize - self.rank)
            self.attn_act_queue = [[[] for _ in range(self.wsize-self.rank)] for _ in range(self.num_fold)]
            self.act_buf_map = {}

    def clean_reqs(self, fold_idx):
        if self.verbose:
            print(f"pp rank {self.rank} fold {fold_idx} clean {len(self.reqs[fold_idx])} reqs", flush=True)

        while self.reqs[fold_idx]:
            req = self.reqs[fold_idx].pop(0)
            req.wait()

    def check_reqs(self):
        for fold in range(self.num_fold):
            assert len(self.reqs[fold]) == 0, \
                f"pp rank {self.rank} fold {fold} has {len(self.reqs[fold])} unfinished reqs"
            
    def check_context(self):
        self.check_reqs()

        assert len(self.labels) == 0, f"pp rank {self.rank} has {len(self.labels)} labels"
        if self.offload_activation:
            assert self.param_batch_idx == 0, \
                f"pp rank {self.rank} has {self.param_batch_idx} param batch idx"
            assert self.attn_batch_idx == self.num_fold * (self.wsize - self.rank), \
                f"pp rank {self.rank} has {self.attn_batch_idx} attn batch idx"
            assert len(self.param_act_queue[0]) == self.param_batch_idx//self.num_fold, \
                f"pp rank {self.rank} has {len(self.param_act_queue[0])} param act queue"
            assert len(self.attn_act_queue[0]) == self.attn_batch_idx//self.num_fold, \
                f"pp rank {self.rank} has {len(self.attn_act_queue[0])} attn act queue"
            
            assert len(self.act_buf_map) == 0, f"pp rank {self.rank} has {len(self.act_buf_map)} act buf map"

            for i in range(0, self.num_fold-1):
                assert len(self.param_act_queue[i]) == len(self.param_act_queue[i+1]), f"pp rank {self.rank} param act queue {i} has {len(self.param_act_queue[i])} elements"
                assert len(self.attn_act_queue[i]) == len(self.attn_act_queue[i+1]), f"pp rank {self.rank} attn act queue {i} has {len(self.attn_act_queue[i])} elements"
        
        for fold in range(self.num_fold):
            for data_offset in range(self.wsize):
                assert len(self.input_tensors[fold][data_offset]) == 0, \
                    f"pp rank {self.rank} fold {fold} data offset {data_offset} has {len(self.input_tensors[fold][data_offset])} input tensors"
                assert len(self.output_tensors[fold][data_offset]) == 0, \
                    f"pp rank {self.rank} fold {fold} data offset {data_offset} has {len(self.output_tensors[fold][data_offset])} output tensors"
            
            assert len(self.reqs[fold]) == 0, \
                f"pp rank {self.rank} fold {fold} has {len(self.reqs[fold])} unfinished reqs"
            assert self.grad_buf[fold] is None, \
                f"pp rank {self.rank} fold {fold} grad buf is not None"

    def deallocate_output_tensor(self, out):
        '''Copied from Megatron
        Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

        This method should be called right after the output tensor has been
        sent to the next pipeline stage. At this point, the output tensor is
        only useful for its '.grad_fn' field, and not its '.data'.
        '''
        if (out is None) or (not self.deallocate_pipeline_outputs):
            return
        assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
        assert out._base is None, "counter-productive to free a view of another tensor."
        out.data = torch.empty((1,), device=out.device, dtype=out.dtype,)

    def register_labels(self, labels):
        self.labels.append(labels)

    def pop_labels(self):
        return self.labels.pop()

    def _get_batch_idx(self, data_repeat_idx, data_offset, fold_idx):
        return data_repeat_idx * self.wsize * self.num_fold +  data_offset * self.num_fold + fold_idx
    
    ##### used by forward pass #####
    # communication primitives
    def recv_ro(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx, 
        r_shape, o_shape, src_tgt, config):
        
        batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
        layer_idx = self.wsize * loop_idx + self.rank

        if r_shape is not None:
            if self.verbose:
                print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                    f"recv (r, o) {batch_idx} from rank {src_tgt}, ", flush=True)
            
            (r, o), reqs = self._communicate(
                config, recv_shapes=[r_shape, o_shape], src_ranks=[src_tgt, src_tgt])

            self.reqs[fold_idx].extend(reqs)
        else:
            assert o_shape is None
            if self.verbose:
                print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                    f"starts batch {batch_idx}", flush=True)
            r, o = None, None
        self.input_tensors[fold_idx][data_offset].append((r, o))

    def recv_ro_send_qkvr(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx, 
        r_shape, o_shape, qkv_tensor, r_tensor, src_tgt, config):
    
        batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
        layer_idx = self.wsize * loop_idx + self.rank
        qkvr_batch_idx = self._get_batch_idx(data_repeat_idx, data_offset-1, fold_idx)

        if r_shape is not None:
            if self.verbose:
                print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                    f"send (qkv, r) {qkvr_batch_idx} to rank {src_tgt}, "
                    f"recv (r, o) {batch_idx} from rank {src_tgt}", flush=True)

            (r, o), reqs = self._communicate(
                config, recv_shapes=[r_shape, o_shape], src_ranks=[src_tgt, src_tgt], 
                send_tensors=[*qkv_tensor, r_tensor], tgt_ranks=[src_tgt for _ in range(len(qkv_tensor)+1)])
        else:
            assert o_shape is None
            if self.verbose:
                print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                    f"send (qkv, r) {qkvr_batch_idx} to rank {src_tgt}, "
                    f"starts batch {batch_idx}", flush=True)
            r, o = None, None
            _, reqs = self._communicate(
                config, send_tensors=[*qkv_tensor, r_tensor], 
                tgt_ranks=[src_tgt for _ in range(len(qkv_tensor)+1)])

        self.input_tensors[fold_idx][data_offset].append((r, o))
        self.reqs[fold_idx].extend(reqs)
    
    def recv_qkvr(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx, 
        qkv_shape, r_shape, src_tgt, config):
        
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank

            print(f"pp rank {self.rank} fold {fold_idx} at ATTN of layer {layer_idx}: "
                f"recv (qkv, r) {batch_idx} from rank {src_tgt}", flush=True)
        
        qkvr, reqs = self._communicate(
            config, recv_shapes=qkv_shape+[r_shape], src_ranks=[src_tgt for _ in range(len(qkv_shape)+1)])

        self.input_tensors[fold_idx][data_offset].append(qkvr)
        self.reqs[fold_idx].extend(reqs)

    def recv_qkvr_send_ro(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx, 
        qkv_shape, r_shape, r_tensor, o_tensor, src_tgt, config):
        
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank
            o_batch_idx = self._get_batch_idx(data_repeat_idx, data_offset+1, fold_idx)

            print(f"pp rank {self.rank} fold {fold_idx} at ATTN of layer {layer_idx}: "
                f"recv (qkv, r) {batch_idx} from rank {src_tgt}, "
                f"send (r, o) {o_batch_idx} to rank {src_tgt}", flush=True)
        
        qkvr, reqs = self._communicate(
            config, recv_shapes=qkv_shape+[r_shape], src_ranks=[src_tgt for _ in range(len(qkv_shape)+1)],
            send_tensors=[r_tensor, o_tensor], tgt_ranks=[src_tgt, src_tgt])

        self.input_tensors[fold_idx][data_offset].append(qkvr)
        self.reqs[fold_idx].extend(reqs)

    def send_ro(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx,
        r_tensor, o_tensor, src_tgt, config):

        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank
        
            print(f"pp rank {self.rank} fold {fold_idx} at ATTN of layer {layer_idx}: "
                f"send (r, o) {batch_idx} to rank {src_tgt}", flush=True)
        
        _, reqs = self._communicate(config=config, send_tensors=[r_tensor, o_tensor], tgt_ranks=[src_tgt, src_tgt])
        self.reqs[fold_idx].extend(reqs)

    # clean reqs and get the input tensors for forward pass
    # these can be merged into a single indexing function
    # but for simplicity and debug, we keep them separate
    def get_ro(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx):
        
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank

            print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: retrieve (r,o) {batch_idx}, "
                f"fold {fold_idx} has reqs {len(self.reqs[fold_idx])}", flush=True)
        
        while self.reqs[fold_idx]:
            req = self.reqs[fold_idx].pop(0)
            req.wait()
        
        return self.input_tensors[fold_idx][data_offset][-1]

    def get_qkvr(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx):

        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = (loop_idx-1)*self.wsize + self.rank + self.wsize-data_offset-1

            print(f"pp rank {self.rank} fold {fold_idx} at ATTN of layer {layer_idx}: retrieve (qkv,r) {batch_idx}, "
                f"fold {fold_idx} has reqs {len(self.reqs[fold_idx])}", flush=True)
        
        while self.reqs[fold_idx]:
            req = self.reqs[fold_idx].pop(0)
            req.wait()
        
        *qkv, r = self.input_tensors[fold_idx][data_offset][-1]
        if data_offset > 0:
            # dereference r
            self.input_tensors[fold_idx][data_offset][-1] = qkv
        return qkv, r

    # register output tensors for backward pass
    # these can be merged into a single append function
    # but for simplicity and debug, we keep them separate
    def register_qkvr(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx, qkv, r):

        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank

            print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                f"output qkvr {batch_idx}", flush=True)
        self.output_tensors[fold_idx][data_offset].append([*qkv, r])

    def register_o(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx, o):
        
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = (loop_idx-1)*self.wsize + self.rank + (self.wsize - data_offset - 1)

            print(f"pp rank {self.rank} fold {fold_idx} at ATTN of layer {layer_idx}: "
                f"output o {batch_idx}", flush=True)
        self.output_tensors[fold_idx][data_offset].append([o,])

    def register_loss(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx, loss):
        
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank

            print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                f"output loss {batch_idx}", flush=True)
        self.output_tensors[fold_idx][data_offset].append([loss,])

    def register_ro(self, data_repeat_idx, loop_idx, data_offset, fold_idx, r, o):
        
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank
        
            print(f"pp rank {self.rank} fold {fold_idx} at ATTN of layer {layer_idx}: "
                f"output ro {batch_idx}", flush=True)
        self.output_tensors[fold_idx][data_offset].append([r, o])

    ##### used by backward pass #####
    # communication primitives
    def send_ro_bwd(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx,
        r_tensor, o_tensor, src_tgt, config):
        
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank

            print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                f"send (r, o) {batch_idx} to rank {src_tgt}", flush=True)
        
        _, reqs = self._communicate(config=config, send_tensors=[r_tensor, o_tensor], tgt_ranks=[src_tgt, src_tgt])
        self.reqs[fold_idx].extend(reqs)
        
    def recv_ro_bwd(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx,
        r_shape, o_shape, src_tgt, config):
        
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = (loop_idx-1)*self.wsize + self.rank + self.wsize-data_offset-1

            print(f"pp rank {self.rank} fold {fold_idx} at ATTN of layer {layer_idx}: "
                f"recv (r, o) {batch_idx} from rank {src_tgt}", flush=True)

        ro, reqs = self._communicate(config, recv_shapes=[r_shape, o_shape], src_ranks=[src_tgt, src_tgt], is_bwd=True, fold=fold_idx)
        self.reqs[fold_idx].extend(reqs)

        assert self.grad_buf[fold_idx] is None
        self.grad_buf[fold_idx] = ro

    def recv_ro_send_qkvr_bwd(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx,
        r_shape, o_shape, qkv_tensor, r_tensor, src_tgt, config):
        
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            rqkv_batch_idx = self._get_batch_idx(data_repeat_idx, data_offset-1, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank

            print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                f"recv (r, o) {batch_idx} from rank {src_tgt}, "
                f"send (qkv, r) {rqkv_batch_idx} to rank {src_tgt}", flush=True)
        
        (r, o), reqs = self._communicate(
            config, send_tensors=qkv_tensor+[r_tensor], tgt_ranks=[src_tgt for _ in range(len(qkv_tensor)+1)],
            recv_shapes=[r_shape, o_shape], src_ranks=[src_tgt, src_tgt], is_bwd=True, fold=fold_idx)
        self.reqs[fold_idx].extend(reqs)

        assert self.grad_buf[fold_idx] is None
        self.grad_buf[fold_idx] = (r, o)

    def recv_qkvr_send_ro_bwd(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx,
        qkv_shape, r_shape, r_tensor, o_tensor, src_tgt, config):

        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            or_batch_idx = self._get_batch_idx(data_repeat_idx, data_offset+1, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank
        
            print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                f"send (r, o) {or_batch_idx} to rank {src_tgt}, "
                f"recv (qkv, r) {batch_idx} from rank {src_tgt}", flush=True)
        
        qkvr, reqs = self._communicate(
            config, recv_shapes=qkv_shape+[r_shape], src_ranks=[src_tgt for _ in range(len(qkv_shape)+1)],
            send_tensors=[r_tensor, o_tensor], tgt_ranks=[src_tgt, src_tgt], is_bwd=True, fold=fold_idx)
        self.reqs[fold_idx].extend(reqs)

        assert self.grad_buf[fold_idx] is None
        self.grad_buf[fold_idx] = qkvr

    def recv_qkvr_bwd(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx,
        qkv_shape, r_shape, src_tgt, config):

        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank

            print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                f"recv (qkv, r) {batch_idx} from rank {src_tgt}", flush=True)
        
        qkvr, reqs = self._communicate(config, recv_shapes=qkv_shape+[r_shape], src_ranks=[src_tgt for _ in range(len(qkv_shape)+1)], is_bwd=True, fold=fold_idx)
        self.reqs[fold_idx].extend(reqs)

        assert self.grad_buf[fold_idx] is None
        self.grad_buf[fold_idx] = qkvr

    def send_qkvr_bwd(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx,
        qkv_tensor, r_tensor, src_tgt, config):

        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = (loop_idx-1)*self.wsize + self.rank + self.wsize-data_offset-1

            print(f"pp rank {self.rank} fold {fold_idx} at ATTN of layer {layer_idx}: "
                f"send (qkv, r) {batch_idx} to rank {src_tgt}", flush=True)
        
        _, reqs = self._communicate(config, send_tensors=qkv_tensor+[r_tensor], tgt_ranks=[src_tgt for _ in range(len(qkv_tensor)+1)])
        self.reqs[fold_idx].extend(reqs)

    # pop input tensors for backward pass
    def pop_input_ro(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx):
        
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank
            print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                f"pop input ro {batch_idx}", flush=True)

        # (r, o)
        ro = self.input_tensors[fold_idx][data_offset].pop()
        return ro
    
    def pop_input_qkvr(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx):
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = (loop_idx-1)*self.wsize + self.rank + self.wsize-data_offset-1

            print(f"pp rank {self.rank} fold {fold_idx} at PARAM+ATTN of layer {layer_idx}: "
                f"pop input qkvr {batch_idx}", flush=True)
        
        return self.input_tensors[fold_idx][data_offset].pop()
    
    def pop_input_qkv(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx):
        
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = (loop_idx-1)*self.wsize + self.rank + self.wsize-data_offset-1

            print(f"pp rank {self.rank} fold {fold_idx} at ATTN of layer {layer_idx}: "
                f"pop input qkv {batch_idx}", flush=True)
        
        return self.input_tensors[fold_idx][data_offset].pop()

    # pop output tensors for backward pass
    def pop_output_loss(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx):
        
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank
        
            print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                f"pop output loss {batch_idx}", flush=True)

        # (loss,)
        return self.output_tensors[fold_idx][data_offset].pop()
    
    def pop_output_qkvr(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx):

        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank

            print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                f"pop output qkvr {batch_idx}", flush=True)

        # (qkv, r)
        return self.output_tensors[fold_idx][data_offset].pop()
    
    def pop_output_o(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx):
        
        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = (loop_idx-1)*self.wsize + self.rank + self.wsize-data_offset-1

            print(f"pp rank {self.rank} fold {fold_idx} at ATTN of layer {layer_idx}: "
                f"pop output o {batch_idx}", flush=True)

        # (o, )
        return self.output_tensors[fold_idx][data_offset].pop()
    
    def pop_output_ro(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx):

        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank

            print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                f"pop output ro {batch_idx}", flush=True)

        # (r, o)
        return self.output_tensors[fold_idx][data_offset].pop()

    # pop gradients w.r.t. output tensors for backward pass
    def pop_grad_qkvr(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx):

        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = self.wsize * loop_idx + self.rank

            print(f"pp rank {self.rank} fold {fold_idx} at PARAM of layer {layer_idx}: "
                f"pop grad qkvr {batch_idx}, wait for {len(self.reqs[fold_idx])} recv reqs", flush=True)
        
        while self.reqs[fold_idx]:
            req = self.reqs[fold_idx].pop(0)
            req.wait()
        
        qkvr_grad = self.grad_buf[fold_idx]
        self.grad_buf[fold_idx] = None

        return qkvr_grad

    def pop_grad_ro(
        self, data_repeat_idx, loop_idx, data_offset, fold_idx):

        if self.verbose:
            batch_idx = self._get_batch_idx(data_repeat_idx, data_offset, fold_idx)
            layer_idx = (loop_idx-1)*self.wsize + self.rank + self.wsize-data_offset-1

            print(f"pp rank {self.rank} fold {fold_idx} at ATTN of layer {layer_idx}: "
                f"pop grad ro {batch_idx}, wait for {len(self.reqs[fold_idx])} recv reqs", flush=True)
        
        while self.reqs[fold_idx]:
            req = self.reqs[fold_idx].pop(0)
            req.wait()
        
        ro = self.grad_buf[fold_idx]
        self.grad_buf[fold_idx] = None

        return ro

    # offload and upload activations
    def push_start(self, activations):
        # activation: [hidden_states, ]
        if self.offload_activation:
            # warmup, work as a single batch
            fold_idx = None
            if self.param_batch_idx < self.num_fold*self.wsize:
                batch_idx_in_loop = self.param_batch_idx%(self.num_fold*self.wsize)
                fold_idx = batch_idx_in_loop % self.num_fold

                self.param_act_queue[fold_idx].append(activations)
                # self.param_act_queue[fold_idx].append([f's_{self.param_batch_idx}'])
                self.param_batch_idx += 1
            else: # steady, work as an appended batch
                param_batch_idx = self.param_batch_idx - 1
                batch_idx_in_loop = param_batch_idx%(self.num_fold*self.wsize)
                fold_idx = batch_idx_in_loop % self.num_fold

                self.param_act_queue[fold_idx][-1].extend(activations)
                # self.param_act_queue[fold_idx][-1].extend([f's_{self.param_batch_idx}'])
            
            if self.offload_verbose:
                act_str = ""
                for idx, each in enumerate(self.param_act_queue[fold_idx][-1]):
                    act_str += f" [{idx}]: {each.view(-1)[0]:.6f}"
                print(f"pp rank {self.rank} fold {fold_idx} param batch counter: {self.param_batch_idx} "
                      f"param act queue size: {len(self.param_act_queue[fold_idx])}, "
                      f"push:{act_str}")

    def push_ro(self, activations):
        # activation: [r, o]
        if self.offload_activation:
            batch_idx_in_loop = self.param_batch_idx%(self.num_fold*self.wsize)
            data_offset = batch_idx_in_loop // self.num_fold
            fold_idx = batch_idx_in_loop % self.num_fold

            r, _ = activations
            if data_offset == 0:
                self.param_act_queue[fold_idx].append([r,])
                # self.param_act_queue[fold_idx].append([f"r_{self.param_batch_idx}"])
            else:
                self.param_act_queue[fold_idx].append(activations)
                # self.param_act_queue[fold_idx].append([f"ro_{self.param_batch_idx}"])
            self.param_batch_idx += 1

            if self.offload_verbose:
                act_str = ""
                for idx, each in enumerate(self.param_act_queue[fold_idx][-1]):
                    act_str += f" [{idx}]: {each.view(-1)[0]:.6f}"
                print(f"pp rank {self.rank} fold {fold_idx} param batch counter: {self.param_batch_idx} offset {data_offset} "
                      f"param act queue size: {len(self.param_act_queue[fold_idx])}, "
                      f"push:{act_str}")
    
    def push_imbo(self, activations):
        # activation: [i, m, b, o]
        if self.offload_activation:
            batch_idx_in_loop = self.attn_batch_idx%(self.num_fold*self.wsize)
            data_offset = self.wsize - batch_idx_in_loop // self.num_fold - 1
            fold_idx = batch_idx_in_loop % self.num_fold

            i, _, _, o = activations
            if data_offset == self.wsize - 1:
                self.attn_act_queue[fold_idx].append([i, o])
                # self.attn_act_queue[fold_idx].append([f"io_{self.attn_batch_idx}"])
            else:
                self.attn_act_queue[fold_idx].append(activations)
                # self.attn_act_queue[fold_idx].append([f"imbo_{self.attn_batch_idx}"])
            self.attn_batch_idx += 1

            if self.offload_verbose:
                act_str = ""
                for idx, each in enumerate(self.attn_act_queue[fold_idx][-1]):
                    act_str += f" [{idx}]: {each.view(-1)[0]:.6f}"
                print(f"pp rank {self.rank} fold {fold_idx} attn batch counter: {self.attn_batch_idx} "
                      f"attn act queue size: {len(self.attn_act_queue[fold_idx])}, "
                      f"push:{act_str}")
    
    def push_end(self, activations):
        if self.offload_activation:
            batch_idx_in_loop = self.param_batch_idx%(self.num_fold*self.wsize)
            data_offset = batch_idx_in_loop // self.num_fold
            fold_idx = batch_idx_in_loop % self.num_fold

            r, _, e = activations
            if data_offset == 0:
                # self.param_act_queue[fold_idx].append([f"re_{self.param_batch_idx}"])
                self.param_act_queue[fold_idx].append([r, e])
            else:
                # self.param_act_queue[fold_idx].append([f"roe_{self.param_batch_idx}"])
                self.param_act_queue[fold_idx].append(activations)
            self.param_batch_idx += 1

            if self.offload_verbose:
                act_str = ""
                for idx, each in enumerate(self.param_act_queue[fold_idx][-1]):
                    act_str += f" [{idx}]: {each.view(-1)[0]:.6f}"
                print(f"pp rank {self.rank} fold {fold_idx} param batch counter: {self.param_batch_idx} "
                      f"param act queue size: {len(self.param_act_queue[fold_idx])}, "
                      f"push:{act_str}")

    def pop_param_act(self):
        if self.offload_activation:
            self.param_batch_idx -= 1

            batch_idx_in_loop = self.param_batch_idx%(self.num_fold*self.wsize)
            # data_offset = batch_idx_in_loop // self.num_fold
            fold_idx = batch_idx_in_loop % self.num_fold

            ret = self.param_act_queue[fold_idx].pop()
            if self.offload_verbose:
                act_str = ""
                for idx, each in enumerate(ret):
                    act_str += f" [{idx}]: {each.view(-1)[0]:.6f}"
                print(f"pp rank {self.rank} fold {fold_idx} param batch counter: {self.param_batch_idx} "
                      f"param act queue size: {len(self.param_act_queue[fold_idx])}, "
                      f"pop:{act_str}")
    
    def pop_start_act(self):
        if self.offload_activation:
            param_batch_idx = self.param_batch_idx - 1

            batch_idx_in_loop = param_batch_idx%(self.num_fold*self.wsize)
            # data_offset = batch_idx_in_loop // self.num_fold
            fold_idx = batch_idx_in_loop % self.num_fold

            ret = self.param_act_queue[fold_idx][-1].pop()
            
            if len(self.param_act_queue[fold_idx][-1]) == 0:
                self.param_act_queue[fold_idx].pop()
                self.param_batch_idx -= 1

            if self.offload_verbose:
                act_str = f" [-1]: {ret.view(-1)[0]:.6f}"
                print(f"pp rank {self.rank} fold {fold_idx} param batch counter: {self.param_batch_idx} "
                      f"param act queue size: {len(self.param_act_queue[fold_idx])}, "
                      f"pop:{act_str}")

    def pop_attn_act(self):
        if self.offload_activation:
            self.attn_batch_idx -= 1

            batch_idx_in_loop = self.attn_batch_idx%(self.num_fold*self.wsize)
            # data_offset = batch_idx_in_loop // self.num_fold
            fold_idx = batch_idx_in_loop % self.num_fold

            ret = self.attn_act_queue[fold_idx].pop()
            if self.offload_verbose:
                act_str = ""
                for idx, each in enumerate(ret):
                    act_str += f" [{idx}]: {each.view(-1)[0]:.6f}"
                print(f"pp rank {self.rank} fold {fold_idx} attn batch counter: {self.attn_batch_idx} "
                      f"attn act queue size: {len(self.attn_act_queue[fold_idx])}, "
                      f"pop:{act_str}")

    def check_act_queue(self):
        if not self.offload_activation:
            return True
        
        param_batch_idx = self.param_batch_idx
        if self.rank == 0:
            param_batch_idx -= self.num_fold * self.wsize

        attn_batch_idx = self.attn_batch_idx
        attn_batch_idx -= self.num_fold * (self.wsize - self.rank)

        assert param_batch_idx == attn_batch_idx, \
            f"pp rank {self.rank} param batch idx {self.param_batch_idx} attn batch idx {self.attn_batch_idx}"
        assert self.param_batch_idx//self.num_fold == len(self.param_act_queue[0]), \
            f"pp rank {self.rank} param batch idx {self.param_batch_idx} param act queue {len(self.param_act_queue[0])}"
        assert self.attn_batch_idx//self.num_fold == len(self.attn_act_queue[0]), \
            f"pp rank {self.rank} param batch idx {self.param_batch_idx} attn act queue {len(self.attn_act_queue[0])}"
        
        for i in range(0, self.num_fold-1):
            assert len(self.param_act_queue[i]) == len(self.param_act_queue[i+1])
            assert len(self.attn_act_queue[i]) == len(self.attn_act_queue[i+1])

    def offload(self):
        if self.offload_activation:
            if self.attn_batch_idx < self.num_fold * self.wsize:
                return
            attn_batch_idx = self.attn_batch_idx

            batch_idx_in_loop = attn_batch_idx % (self.num_fold*self.wsize)
            data_offset = batch_idx_in_loop // self.num_fold
            fold_idx = batch_idx_in_loop % self.num_fold
            
            attn_data_idx = -data_offset*2 - 1
            param_data_idx = -(self.wsize - data_offset)
            attn_data = self.attn_act_queue[fold_idx][attn_data_idx]
            param_data = self.param_act_queue[fold_idx][param_data_idx]
            
            if self.verbose:
                print(f"pp rank {self.rank} fold {fold_idx} offload attn batch {attn_batch_idx} "
                    f"data offset {attn_data_idx} param data offset {param_data_idx} ")
                    #   f"attn data {attn_data} param data {param_data}")
            
            # get host memory buffer
            offload_data = attn_data + param_data
            shape = tuple([tuple(each.shape) for each in offload_data])

            if shape not in self.free_list:
                self.free_list[shape] = []
            if len(self.free_list[shape]) == 0:
                buffer = []
                for each in offload_data:
                    buffer.append(torch.empty_like(each, device='cpu', pin_memory=True).untyped_storage())
                self.free_list[shape].append(buffer)
            buffer = self.free_list[shape].pop()

            # actual offload
            self.offload_stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(self.offload_stream):
                for act, buf in zip(offload_data, buffer):
                    assert act not in self.act_buf_map
                    self.act_buf_map[act] = buf
                    buf.copy_(act.untyped_storage(), non_blocking=True)
                    act.record_stream(self.offload_stream)
                    act.untyped_storage().resize_(0)

    def upload(self):
        if self.offload_activation:
            attn_batch_idx = self.attn_batch_idx - 1
            if attn_batch_idx < self.num_fold * self.wsize:
                return
            batch_idx_in_loop = attn_batch_idx % (self.num_fold*self.wsize)
            data_offset = batch_idx_in_loop // self.num_fold
            fold_idx = batch_idx_in_loop % self.num_fold

            attn_data_idx = -data_offset*2 - 2
            param_data_idx = -(self.wsize - data_offset)
            attn_data = self.attn_act_queue[fold_idx][attn_data_idx]
            param_data = self.param_act_queue[fold_idx][param_data_idx]

            if self.verbose:
                print(f"pp rank {self.rank} fold {fold_idx} upload attn batch idx {attn_batch_idx} "
                    f"data offset {attn_data_idx} param data offset {param_data_idx} ")
                    #   f"attn data {attn_data} param data {param_data}")
            
            # get host memory buffer
            offload_data = attn_data + param_data
            shape = tuple([tuple(each.shape) for each in offload_data])

            buffer = []
            for each in offload_data:
                buffer.append(self.act_buf_map.pop(each))
            
            # actual upload
            self.offload_stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(self.offload_stream):
                for act, buf in zip(attn_data+param_data, buffer):
                    act.untyped_storage().resize_(act.nbytes)
                    act.untyped_storage().copy_(buf, non_blocking=True)

            # reuse the host memory buffer
            assert shape in self.free_list
            self.free_list[shape].append(buffer)


def get_pipeline_context():
    global _GLOBAL_PIPELINE_CONTEXT
    assert _GLOBAL_PIPELINE_CONTEXT is not None
    return _GLOBAL_PIPELINE_CONTEXT


def set_pipeline_context(args):
    if args.attention_pipeline:
        global _GLOBAL_PIPELINE_CONTEXT
        _GLOBAL_PIPELINE_CONTEXT = PipelineContext(args)
