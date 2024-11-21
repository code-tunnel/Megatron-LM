import sys
sys.path.append(".") # enter from Megatron-LM
import os

import torch
import torch.distributed as dist

from megatron import get_args, get_timers
from megatron.arguments import core_transformer_config_from_args
from megatron.initialize import initialize_megatron, set_jit_fusion_options
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, _set_cuda_rng_state
from megatron.model.transformer import ParallelTransformerLayer
from megatron.model.enums import AttnMaskType, LayerType
from megatron.model import Float16Module, DistributedDataParallel as LocalDDP
from megatron.optimizer import get_megatron_optimizer

from megatron.model.flatten_transformer_layer import FlattenTransformerLayer
from megatron.model.customized_modules import CustomizedLayer


@torch.no_grad()
def get_diff(a, b):
    diff = torch.abs(a- b)
    max_diff = torch.max(diff)
    min_diff = torch.min(diff)
    return max_diff.item(), min_diff.item()


def sanity_check(args, config, device, rank, world_size, timers):
    # reference model
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
    ref_model = ParallelTransformerLayer(
        config,
        layer_number=1,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.causal,
    ).cuda(device)
    if args.fp16 or args.bf16:
        ref_model = Float16Module(ref_model, args)
    ref_model = LocalDDP(ref_model,
                     args.accumulate_allreduce_grads_in_fp32, 
                     args.use_contiguous_buffers_in_local_ddp)
    
    # current model
    torch.set_rng_state(cpu_rng_state)
    _set_cuda_rng_state(cuda_rng_state)
    get_cuda_rng_tracker().set_states(cuda_rng_state_tracker)
    model = CustomizedLayer(
        config,
        layer_number=1,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.causal,
    ).cuda(device)
    if args.fp16 or args.bf16:
        model = Float16Module(model, args)
    model = LocalDDP(model,
                     args.accumulate_allreduce_grads_in_fp32, 
                     args.use_contiguous_buffers_in_local_ddp)
    
    # optimizer
    ref_optimizer = get_megatron_optimizer([ref_model])
    optimizer = get_megatron_optimizer([model])

    original_params = []
    # sanity check for parameter initialization
    for (ref_name, ref_param), (cur_name, cur_param) in \
        zip(ref_model.named_parameters(), model.named_parameters()):
        ref = '.'.join(ref_name.split('.')[-2:])
        cur = '.'.join(cur_name.split('.')[-2:])
        assert ref == cur
        if 'bias' in cur_name or 'layernorm' in cur_name:
            assert torch.allclose(ref_param, cur_param), \
                f"param name: {cur_name}, shape: {cur_param.shape}"
        else:
            # test random init parameters
            assert torch.allclose(ref_param, cur_param), \
                f"param name: {cur_name}, shape: {cur_param.shape}"
        original_params.append((cur_name, cur_param.detach().clone()))
    
    if rank == 0:
        print("pass parameter initialization")
    
    # prepare data
    bs = 1
    hs = args.hidden_size
    sl = args.seq_length
    dt = torch.float16
    local_sl = sl
    if args.sequence_parallel:
        local_sl = sl//world_size
    input_tensor = torch.rand(
        [local_sl, bs, hs], device=device, requires_grad=True, dtype=dt)
    ref_input_tensor = input_tensor.detach().clone().requires_grad_(True)
    grad_tensor = torch.rand(
        [local_sl, bs, hs], device=device, dtype=dt)

    # sanity check for forward pass
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
    ref_output_tensor = ref_model(ref_input_tensor, None)

    torch.set_rng_state(cpu_rng_state)
    _set_cuda_rng_state(cuda_rng_state)
    get_cuda_rng_tracker().set_states(cuda_rng_state_tracker)
    output_tensor = model(input_tensor, None)

    assert torch.allclose(ref_output_tensor, output_tensor), \
        f"forward output diff: {get_diff(ref_output_tensor, output_tensor)}"
    if rank == 0:
        print("pass forward pass")

    # sanity check for backward pass
    ref_output_tensor.backward(grad_tensor)
    output_tensor.backward(grad_tensor)

    for (ref_name, ref_param), (cur_name, cur_param) in \
        zip(ref_model.named_parameters(), model.named_parameters()):
        ref = '.'.join(ref_name.split('.')[-2:])
        cur = '.'.join(cur_name.split('.')[-2:])
        assert ref == cur

        assert cur_param.main_grad is not None, f"{cur_name} grad is None"
        assert ref_param.main_grad is not None, f"{ref_name} grad is None"

        # params' grad before flash attn kernel have difference
        # This is caused by flash attn kernel, which is not reproducible and rng state cannot control.
        # I suggest replacing official flash attn kernel with triton's to ensure reproducibility
        if rank == 0:
            print(f"name: {cur_name}, max grad: {torch.max(cur_param.main_grad).item()}, grad diff: {get_diff(ref_param.main_grad, cur_param.main_grad)}")

        # assert torch.allclose(ref_param.main_grad, cur_param.main_grad), \
        #     f"{cur_name} has grad difference, {get_diff(ref_param.main_grad, cur_param.main_grad)}"

    print(f"grad diff: {get_diff(ref_input_tensor.grad, input_tensor.grad)}")
    # assert torch.allclose(ref_input_tensor.grad, input_tensor.grad, atol=1e-5, rtol=1e-3), \
    #     f"{get_diff(ref_input_tensor.grad, input_tensor.grad)}"
    if rank == 0:
        print("(Do I?) pass backward pass")
    
    # sanity check for optimizer
    ref_success, _, _ = ref_optimizer.step(args, timers)
    cur_success, _, _ = optimizer.step(args, timers)

    if rank == 0:  
        print(f"ref success? {ref_success}, cur success? {cur_success}")

    for (ref_name, ref_param), (cur_name, cur_param), (org_name, org_param) in \
        zip(ref_model.named_parameters(), model.named_parameters(), original_params):
        ref = '.'.join(ref_name.split('.')[-2:])
        cur = '.'.join(cur_name.split('.')[-2:])
        assert ref == cur

        # to pass this assertion, try to tune loss scale and lr
        assert not torch.allclose(org_param, cur_param), f"{cur_name} is not changed"

        # assert torch.allclose(ref_param, cur_param, atol=1e-3, rtol=1e-4), \
        #         f"param name: {cur_name}, shape: {cur_param.shape}, max diff: {torch.max(torch.abs(ref_param-cur_param))}"
        print(f"param name: {cur_name}, shape: {cur_param.shape}, max diff: {torch.max(torch.abs(ref_param-cur_param))}")
    
    if rank == 0:
        print("(Do I?) pass optimizer")
    exit(0)


def memory_check(args, config, device, rank, world_size, timers):
    # torch.cuda.memory._record_memory_history()

    # prepare data
    bs = 1
    hs = args.hidden_size
    sl = args.seq_length
    dt = torch.float16
    local_sl = sl
    if args.sequence_parallel:
        local_sl = sl//world_size
    
    model = ParallelTransformerLayer(
        config,
        layer_number=1,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.causal,
        drop_path_rate=0.
    ).cuda(device)
    if args.fp16 or args.bf16:
        model = Float16Module(model, args)
    model = LocalDDP(model,
                     args.accumulate_allreduce_grads_in_fp32, 
                     args.use_contiguous_buffers_in_local_ddp)

    model_params = hs / world_size * (12*hs+7+6*world_size)

    optimizer = get_megatron_optimizer([model])
    # 1 grad (buffer in LocalDDP), 2 fp32 copy
    # 2 adam params will be intialized only when .step is invoked
    optimizer_params = (1 + 2) * model_params

    # total = 0
    # for s in optimizer.optimizer.state:
    #     total += s.numel()
    #     print(f"{optimizer.optimizer.state[s]}")
    # print(f"model_param: {model_params}, state: {total}")

    if rank == 0:
        estimated = (model_params + optimizer_params) * 2
        actual = torch.cuda.memory_allocated()
        print(f"estimated: {estimated}, actual: {actual}, diff: {actual - estimated}")

    input_tensor = torch.rand(
        [local_sl, bs, hs], device=device, requires_grad=True, dtype=dt)
    grad_tensor = torch.rand(
        [local_sl, bs, hs], device=device, dtype=dt)
    
    activation_params = local_sl*bs*hs
    if rank == 0:
        # this remains as global memory buffer for later memory computation
        estimated += activation_params * 2 * 2
        actual = torch.cuda.memory_allocated()
        print(f"estimated: {estimated}, actual: {actual}, diff: {actual - estimated}")
    
    output = model(input_tensor, None)
    if rank == 0:
        cur = estimated + activation_params * (1+2) * 2
        actual = torch.cuda.memory_allocated()
        print(f"forward pass: {actual}, cur: {cur}, diff: {(actual-cur)/1024**2}")
    
    output.backward(grad_tensor)
    if rank == 0:
        actual = torch.cuda.memory_allocated()
        print(f"backward pass: {actual}")
    
    optimizer.step(args, timers)
    if rank == 0:
        actual = torch.cuda.memory_allocated()
        print(f"optimizer: {actual}")

    optimizer.zero_grad()
    # del input_tensor
    # del grad_tensor
    # del output

    if rank == 0:
        estimated += 2*model_params*4
        actual = torch.cuda.memory_allocated()
        print(f"clean: {actual}, estimated: {estimated}, {(actual-estimated)/1024**2}")
    # torch.cuda.memory._dump_snapshot(f"layer_snapshot")

    exit(0)


def main(extra_args_provider=None, args_defaults={}):
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    args = get_args()
    timers = get_timers()
    config = core_transformer_config_from_args(args)
    device = torch.cuda.current_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    print(f"rank: {rank}/{world_size}, tp_rank: {tp_rank}/{tp_size}")
    sanity_check(args, config, device, rank, world_size, timers)
    # memory_check(args, config, device, rank, world_size, timers)


if __name__ == "__main__":
    main(args_defaults={'tokenizer_type': 'GPT2BPETokenizer', 
                        'vocab_file':'scripts/gpt2tokenizer/gpt2-vocab.json',
                        'merge_file':'scripts/gpt2tokenizer/gpt2-merges.txt',
                       })
