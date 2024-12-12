import sys
sys.path.append(".") # enter from Megatron-LM
import os
import numpy as np
import torch
import torch.distributed as dist

from megatron import get_args
from megatron.arguments import core_transformer_config_from_args
from megatron.initialize import initialize_megatron, set_jit_fusion_options
from megatron.model.transformer import ParallelTransformerLayer, get_layer_fwd_start, get_layer_fwd_end, \
    get_layer_bwd_start, get_layer_bwd_end, get_attn_fwd_start, get_attn_fwd_end, get_attn_bwd_start, get_attn_bwd_end
from megatron.model.flatten_transformer_layer import FlattenTransformerLayer
from megatron.core.tensor_parallel import get_cuda_rng_tracker, checkpoint
from megatron.model.enums import AttnMaskType, LayerType
from megatron.model import Float16Module, DistributedDataParallel as LocalDDP


def init(module, input_tensor, grad_tensor, rng_tracker):
    input_tensor.grad = None
    module.zero_grad()
    with rng_tracker.fork():
        torch.nn.init.uniform_(input_tensor)
        torch.nn.init.uniform_(grad_tensor)


def forward(module, input_tensor):
    output_tensor = module(input_tensor, None)
    # output_tensor = checkpoint(
    #     lambda _x: module(_x, None),
    #     False,
    #     input_tensor)
    return output_tensor


def backward(output_tensor, grad_tensor):
    output_tensor.backward(grad_tensor)


def step(module, input_tensor, grad_tensor):
    with torch.profiler.record_function(f"Megatron FWD"):
        output_tensor = module(input_tensor, None)
        # output_tensor = checkpoint(
        #     lambda _x: module(_x, None),
        #     False,
        #     input_tensor)
    
    with torch.profiler.record_function(f"Megatron BWD"):
        output_tensor.backward(grad_tensor)


def main(run_benchmark, run_profile, 
         extra_args_provider=None, args_defaults={}):
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    args = get_args()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    rng_tracker = get_cuda_rng_tracker()
    device = torch.cuda.current_device()

    fmt_str = "batch size: {}, seq len: {}, hidden size: {}"

    bs = args.micro_batch_size
    sl = args.seq_length
    hs = args.hidden_size
    dt = torch.float16
    
    local_sl = sl
    if args.sequence_parallel:
        local_sl = sl//world_size
    input_tensor = torch.empty(
        [local_sl, bs, hs], device=device, requires_grad=True, dtype=dt)
    grad_tensor = torch.empty(
        [local_sl, bs, hs], device=device, requires_grad=True, dtype=dt)

    config = core_transformer_config_from_args(args)

    if args.checkpoint_without_attn:
        layer_module = FlattenTransformerLayer
    else:
        layer_module = ParallelTransformerLayer
    model = layer_module(
        config,
        layer_number=1,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.causal,
        drop_path_rate=0.
    ).cuda(device)
    
    if args.fp16 or args.bf16:
        model = Float16Module(model, args)
    model = LocalDDP(
        model, args.accumulate_allreduce_grads_in_fp32, 
        args.use_contiguous_buffers_in_local_ddp)
    
    for _ in range(5):
        init(model, input_tensor, grad_tensor, rng_tracker)
        output_tensor = forward(model, input_tensor)
        backward(output_tensor, grad_tensor)
    
    pre_attn_fwd, attn_fwd, post_attn_fwd = [], [], []
    pre_attn_bwd, attn_bwd, post_attn_bwd = [], [], []
    layer_fwd = 0
    for _ in range(5):
        init(model, input_tensor, grad_tensor, rng_tracker)
        output_tensor = forward(model, input_tensor)
        backward(output_tensor, grad_tensor)

        assert get_layer_fwd_start() != layer_fwd
        layer_fwd = get_layer_fwd_start()

        pre_attn_fwd.append(get_attn_fwd_start() - get_layer_fwd_start())
        attn_fwd.append(get_attn_fwd_end() - get_attn_fwd_start())
        post_attn_fwd.append(get_layer_fwd_end() - get_attn_fwd_end())

        post_attn_bwd.append(get_attn_bwd_start() - get_layer_bwd_start())
        attn_bwd.append(get_attn_bwd_end() - get_attn_bwd_start())
        pre_attn_bwd.append(get_layer_bwd_end() - get_attn_bwd_end())
    if rank == 0:
        print(f"data shape: {input_tensor.shape}")
        print(f"pre attn fwd: {np.mean(pre_attn_fwd):.4f} +- {np.std(pre_attn_fwd):.4f}"
              f"\nattn fwd: {np.mean(attn_fwd):.4f} +- {np.std(attn_fwd):.4f}"
              f"\npost attn fwd: {np.mean(post_attn_fwd):.4f} +- {np.std(post_attn_fwd):.4f}"
              f"\npre attn bwd: {np.mean(pre_attn_bwd):.4f} +- {np.std(pre_attn_bwd):.4f}"
              f"\nattn bwd: {np.mean(attn_bwd):.4f} +- {np.std(attn_bwd):.4f}"
              f"\npost attn bwd: {np.mean(post_attn_bwd):.4f} +- {np.std(post_attn_bwd):.4f}")


if __name__ == "__main__":    
    run_benchmark = int(os.getenv("BENCHMARK", str(0)))
    run_profile = int(os.getenv("PROFILE", str(0)))

    main(run_benchmark, run_profile,
         args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                        'vocab_file':'scripts/gpt2tokenizer/gpt2-vocab.json',
                        'merge_file':'scripts/gpt2tokenizer/gpt2-merges.txt',
                       })
