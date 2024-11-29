from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import detach_variable
from einops import rearrange
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, \
    _flash_attn_varlen_backward, flash_attn_varlen_func

from megatron import get_args, core, get_timers, get_tokenizer
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args
from megatron.core import tensor_parallel, parallel_state as mpu
from megatron.core.tensor_parallel.mappings import _reduce_scatter_along_first_dim
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker,\
    _set_cuda_rng_state
from megatron.core.pipeline_parallel import get_pipeline_context
from megatron.model import LayerNorm
from megatron.model.module import MegatronModule
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.triton_bias_gelu import fused_bias_gelu_triton, _fused_bias_gelu_fwd_kernel, _fused_bias_gelu_bwd_kernel
from megatron.model.transformer import get_bias_dropout_add, \
            bias_dropout_add_fused_train, bias_dropout_add_fused_inference
from megatron.model.language_model import Embedding

import fused_weight_gradient_mlp_cuda


class GradPatch(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, weight, bias):
        return weight.detach().requires_grad_(weight.requires_grad), \
            bias.detach().requires_grad_(bias.requires_grad)

    @staticmethod
    def backward(ctx, weight_grad, bias_grad):
        return weight_grad, bias_grad


class PreAttnModule(MegatronModule):
    def __init__(self, config, args,
                 layer_number, self_attn_mask_type=AttnMaskType.padding,):
        
        super(PreAttnModule, self).__init__()
        self.apply_residual_connection_post_layernorm \
            = config.apply_residual_connection_post_layernorm
        
        ############### kernel 1 ###############
        # Layernorm 1 on the input data.
        self.input_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=config.sequence_parallel,
            apply_layernorm_1p=args.apply_layernorm_1p)

        ############### kernel 2 ###############
        # Self attention - QKV Linear
        query_projection_size, kv_projection_size = self._add_attention_block_attrs(
            args, config, layer_number, AttnType.self_attn, self_attn_mask_type)
        self.query_key_value = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                query_projection_size + 2 * kv_projection_size,
                config=config,
                init_method=config.init_method,
                bias=args.add_bias_linear,
                gather_output=False)
        
        self.transfer_weight = args.transfer_weight

        self.chunk_size = args.chunk_size
    
    def _add_attention_block_attrs(self, args, config, 
                                   layer_number, attention_type, attn_mask_type):
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = config.params_dtype
        self.sequence_parallel = config.sequence_parallel
        self.group_query_attention = args.group_query_attention
        self.num_query_groups = args.num_query_groups

        query_projection_size = config.kv_channels * config.num_attention_heads
        if self.group_query_attention:
            kv_projection_size = args.kv_channels * args.num_query_groups
        else:
            kv_projection_size = args.kv_channels * args.num_attention_heads

        self.use_flash_attn = args.use_flash_attn \
            and attention_type == AttnType.self_attn \
            and self.attn_mask_type == AttnMaskType.causal
        assert self.use_flash_attn, "must enable --use-flash-attn"

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = core.utils.divide(
            query_projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            config.num_attention_heads, world_size)

        if self.group_query_attention:
            if args.num_query_groups % world_size != 0:
                raise NotImplementedError('Currently the num_query_groups should be '
                                          'a multiple of the tensor parallel size')
            self.num_query_groups_per_partition = core.utils.divide(
                        args.num_query_groups, world_size)
        else:
            self.num_query_groups_per_partition = self.num_attention_heads_per_partition

        return query_projection_size, kv_projection_size

    def forward(self, hidden_states):
        # hidden_states: [s/t, b, h]
        ############### kernel 1 ###############
        # Layer norm at the beginning of the transformer layer.
        if self.chunk_size > 0:
            hidden_state_splits = torch.split(hidden_states, self.chunk_size, dim=0)
            layernorm_output = []
            for split in hidden_state_splits:
                layernorm_output.append(self.input_layernorm(split))
            layernorm_output = torch.cat(layernorm_output, dim=0)
        else:
            layernorm_output = self.input_layernorm(hidden_states)

        # prepare for kernel 4
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        
        if self.transfer_weight:
            weight, bias = GradPatch.apply(self.query_key_value.weight, self.query_key_value.bias)
            # [s/t, b, h], [3h/t, h], [3h/t,], [s/t, b, h]
            return layernorm_output, weight, bias, residual
        else:
            # Self attention.
            ############### kernel 2 ###############
            # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
            # here, np = ng
            mixed_x_layer, _ = self.query_key_value(layernorm_output)
            # [s, b, 3h/t], [s/t, b, h]
            return mixed_x_layer, residual


class ChunkedMLP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        layernorm_output, dense_h_to_4h_weight, dense_h_to_4h_bias, 
        dense_4h_to_h_weight, dense_4h_to_h_bias, 
        chunk_size, sequence_parallel,):
        
        # save for linear 1
        saved_tensors = [layernorm_output, dense_h_to_4h_weight, dense_h_to_4h_bias, dense_4h_to_h_weight]
        ############### Linear 1 ################
        # the gathered layernorm_output should be allocated by global buffer which won't cause fragmentation
        # layernorm_output: s/t, b, h -> s, b, h
        if sequence_parallel:
            world_size = mpu.get_tensor_model_parallel_world_size()
            dim_size = list(layernorm_output.size())
            dim_size[0] = dim_size[0] * world_size
            total_input = mpu.get_global_memory_buffer().get_tensor(dim_size, layernorm_output.dtype, "mpu")
            torch.distributed._all_gather_base(
                total_input, layernorm_output, group=mpu.get_tensor_model_parallel_group()
            )
        else:
            total_input = layernorm_output

        # s, b, 4h/t
        # output_ = torch.matmul(total_input, dense_h_to_4h_weight.t())

        ############### GeLU ################
        # output_ = fused_bias_gelu_triton(output_, dense_h_to_4h_bias)

        ############### Linear 2 ################
        # s, b, 4h/t -> s, b, h
        # output_ = torch.matmul(output_, dense_4h_to_h_weight.t())

        intermediate_shape = list(total_input.size())
        intermediate_shape[-1] = dense_h_to_4h_weight.size(0)
        intermediate_splits = []
        linear2_input_splits = []

        # s, b, h
        if sequence_parallel:
            output_before_reduce_scatter = mpu.get_global_memory_buffer().get_tensor(list(total_input.size()), layernorm_output.dtype, "full_seq")
        else:
            output_before_reduce_scatter = torch.empty_like(total_input)
        # [s/c, b, h]
        input_splits = torch.split(total_input, chunk_size, dim=0)
        output_splits_before_reduce_scatter = torch.split(output_before_reduce_scatter, chunk_size, dim=0)
        assert len(input_splits) == len(output_splits_before_reduce_scatter), f"len: {len(input_splits)} - {len(output_splits_before_reduce_scatter)}"

        num_splits = len(input_splits)
        N = dense_h_to_4h_weight.size(0)
        for i in range(num_splits):
            # chunked linear 1
            # s/c, b, 4h/t
            _split = torch.matmul(input_splits[i], dense_h_to_4h_weight.t())
            intermediate_splits.append(_split)

            # chunked gelu fusion
            _output = torch.empty_like(_split)
            _output_arg = _output.view(-1, N)
            _input_arg = _split.view(-1, N)
            M = _input_arg.shape[0]
            _fused_bias_gelu_fwd_kernel[(M,)](
                dense_h_to_4h_bias, _input_arg, _output_arg,
                _input_arg.stride(0), N, BLOCK_SIZE=1024)

            linear2_input_splits.append(_output)
            # chunked linear 2
            # s/c, b, h
            torch.matmul(_output, dense_4h_to_h_weight.t(), out=output_splits_before_reduce_scatter[i])

        # save for gelu fusion
        saved_tensors.extend(intermediate_splits)
        
        # save for linear 2
        saved_tensors.extend(linear2_input_splits)

        # s, b, h -> s/t, b, h
        if sequence_parallel:
            _output = _reduce_scatter_along_first_dim(output_before_reduce_scatter)
        else:
            _output = output_before_reduce_scatter
        
        output = _output.add_(dense_4h_to_h_bias)

        ctx.save_for_backward(*saved_tensors)
        ctx.num_splits = num_splits
        ctx.chunk_size = chunk_size
        ctx.sequence_parallel = sequence_parallel
        return output

    @staticmethod
    def backward(ctx, grad_output):
        sequence_parallel = ctx.sequence_parallel
        layernorm_output, dense_h_to_4h_weight, dense_h_to_4h_bias, dense_4h_to_h_weight, *act_splits = ctx.saved_tensors
        intermediate_splits = act_splits[:ctx.num_splits]
        linear2_input_splits = act_splits[ctx.num_splits:]

        ############### Linear 2 ################
        # s/t, b, h -> s, b, h
        if sequence_parallel:
            world_size = mpu.get_tensor_model_parallel_world_size()
            dim_size = list(grad_output.size())
            dim_size[0] = dim_size[0] * world_size
            total_grad_output = mpu.get_global_memory_buffer().get_tensor(dim_size, grad_output.dtype, "mpu")
            torch.distributed._all_gather_base(
                total_grad_output, grad_output, group=mpu.get_tensor_model_parallel_group()
            )
        else:
            total_grad_output = grad_output
        
        local_bs, hidden_size = total_grad_output.shape[-2:]
        intermediate_hidden_size = dense_h_to_4h_weight.size(0)

        # s/c, b, h
        total_grad_output_splits = torch.split(total_grad_output, ctx.chunk_size, dim=0)
        assert len(total_grad_output_splits) == ctx.num_splits, f"len: {len(total_grad_output_splits)} - {ctx.num_splits}"
        # grad_dense_4h_to_h_weight = total_grad_output.t().matmul(linear2_input)
        # gradient_accumulation_fusion by default is True
        for i in range(1):
            # s/c * b, 4h/t
            # linear2_input_split = linear2_input_splits[i].view(-1, intermediate_hidden_size)
            # s/c * b, h
            # total_grad_output_split = total_grad_output_splits[i].view(-1, hidden_size)

            linear2_input = torch.cat(linear2_input_splits, dim=0)
            if dense_4h_to_h_weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    linear2_input, total_grad_output, dense_4h_to_h_weight.main_grad
                )
            elif dense_4h_to_h_weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    linear2_input, total_grad_output, dense_4h_to_h_weight.main_grad
                )
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
        grad_dense_4h_to_h_weight = None
        grad_dense_4h_to_h_bias = total_grad_output.sum(0)

        # s, b, 4h/t
        # grad_linear2_input = torch.matmul(total_grad_output, dense_4h_to_h_weight)

        ############### GeLU ################
        grad_intermediate_splits = []
        N = intermediate_hidden_size
        # grad_dense_h_to_4h_bias = torch.zeros_like(dense_h_to_4h_bias)
        for i in range(ctx.num_splits):
            # chunked linear 2
            # s/c, b, 4h/t
            grad_linear2_input_split = torch.matmul(total_grad_output_splits[i], dense_4h_to_h_weight)

            _input = intermediate_splits[i]
            grad_input = torch.empty_like(_input)

            _input_arg = _input.view(-1, N)
            M = _input_arg.shape[0]
            grad_input_arg = grad_input.view(-1, N)

            _fused_bias_gelu_bwd_kernel[(M,)](
                grad_linear2_input_split, _input_arg, dense_h_to_4h_bias, grad_input_arg,
                _input_arg.stride(0), N, BLOCK_SIZE=1024)

            grad_intermediate_splits.append(grad_input)
            # manually accumulate bias gradient
            # grad_dense_h_to_4h_bias.add_(torch.sum(grad_input_arg, dim=0))
        
        ############### Linear 1 ################
        # s/t, b, h -> s, b, h
        # should equivalent to total_grad_output
        if ctx.sequence_parallel:
            world_size = mpu.get_tensor_model_parallel_world_size()
            dim_size = list(layernorm_output.size())
            dim_size[0] = dim_size[0] * world_size
            total_input = mpu.get_global_memory_buffer().get_tensor(dim_size, layernorm_output.dtype, "mpu")
            handle = torch.distributed._all_gather_base(
                total_input, layernorm_output, group=mpu.get_tensor_model_parallel_group(), async_op=True
            )
        else:
            total_input = layernorm_output
        
        # s, b, h
        if ctx.sequence_parallel:
            grad_input = mpu.get_global_memory_buffer().get_tensor(list(total_input.shape), total_input.dtype, "full_seq")
        else:
            grad_input = torch.empty_like(total_input)
        grad_input_splits = torch.split(grad_input, ctx.chunk_size, dim=0)
        assert len(grad_input_splits) == len(intermediate_splits), f"len: {len(grad_input_splits)} - {len(intermediate_splits)}"

        for i in range(ctx.num_splits):
            # s/c, b, h
            torch.matmul(grad_intermediate_splits[i], dense_h_to_4h_weight, out=grad_input_splits[i])
        
        if ctx.sequence_parallel:
            handle.wait()

        if ctx.sequence_parallel:
            dim_size = list(layernorm_output.size())
            sub_grad_input = torch.empty(dim_size, dtype=layernorm_output.dtype, device=layernorm_output.device)
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input, grad_input, group=mpu.get_tensor_model_parallel_group(), async_op=True
            )

        # s/c, b, h
        total_input_splits = torch.split(total_input, ctx.chunk_size, dim=0)
        assert len(total_input_splits) == ctx.num_splits, f"len: {len(total_input_splits)} - {ctx.num_splits}"

        # grad_weight = grad_output.t().matmul(total_input)
        for i in range(1):
            # total_input_split = total_input_splits[i].view(-1, hidden_size)
            # grad_intermediate_split = grad_intermediate_splits[i].view(-1, intermediate_hidden_size)
            grad_intermediate = torch.cat(grad_intermediate_splits, dim=0)
            grad_dense_h_to_4h_bias = grad_intermediate.view(-1, intermediate_hidden_size).sum(0)
            if dense_h_to_4h_weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    total_input, grad_intermediate, dense_h_to_4h_weight.main_grad
                )

            elif dense_h_to_4h_weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    total_input, grad_intermediate, dense_h_to_4h_weight.main_grad
                )
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
        grad_dense_h_to_4h_weight = None
        
        if ctx.sequence_parallel:
            handle.wait()
            return (
                sub_grad_input, grad_dense_h_to_4h_weight, grad_dense_h_to_4h_bias,
                grad_dense_4h_to_h_weight, grad_dense_4h_to_h_bias, None, None
            )
        
        return (
            grad_input, grad_dense_h_to_4h_weight, grad_dense_h_to_4h_bias,
            grad_dense_4h_to_h_weight, grad_dense_4h_to_h_bias, None, None
        )


class PostAttnModule(MegatronModule):
    def __init__(self, config, args, layer_number):

        super(PostAttnModule, self).__init__()
        self.layer_number = layer_number
        self.apply_residual_connection_post_layernorm \
            = config.apply_residual_connection_post_layernorm
        
        ############### kernel 4 ################
        # Self attention - O Linear
        self.dense = tensor_parallel.RowParallelLinear(
            config.kv_channels * config.num_attention_heads, # query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=args.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True)
        
        # hidden dropout is appended to O Linear
        self.hidden_dropout = config.hidden_dropout
        self.bias_dropout_fusion = config.bias_dropout_fusion
        self.drop_path = None

        ############### MLP BLOCK ################
        ############### kernel 5 ################
        # Layernorm 2 on the attention output
        self.post_attention_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            no_persist_layer_norm=not config.persist_layer_norm,
            sequence_parallel=config.sequence_parallel,
            apply_layernorm_1p=args.apply_layernorm_1p)
        
        ############### kernel 6 ################
        # MLP - Linear 1
        self._add_mlp_block_attrs(args, config)
        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=self.add_bias,
            gather_output=False,
            skip_bias_add=True,
        )

        ############### kernel 7 ################
        # MLP - GeLU
        self.use_triton_fusion = args.use_triton_fusion
        
        ############### kernel 8 ################
        # MLP - Linear 2
        # Project back to h.
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=self.add_bias,
            input_is_parallel=True
        )

        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
                nullcontext if use_nvfuser else torch.enable_grad

        self.retriever = None

        self.chunk_size = args.chunk_size
        self.sequence_parallel = config.sequence_parallel
    
    def _add_mlp_block_attrs(self, args, config):
        self.add_bias = config.add_bias_linear

        assert config.ffn_hidden_size // 4 == config.hidden_size, \
            f"only support gelu for now"
        
        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        self.swiglu = args.swiglu

    def forward(self, context_layer, residual):
        # context_layer: [s*b, a/t, d], residual: [s/t, b, h]
        batch_size = residual.shape[1]
        # s*b, a/t, d -> s, b, h/t
        context_layer = rearrange(context_layer, '(b s) h d -> s b (h d)', b=batch_size)

        # this is wrong when using sp & tp
        # if self.chunk_size > 0:
        #     context_layer_splits = torch.split(context_layer, self.chunk_size, dim=0)
        #     residual_splits = torch.split(residual, self.chunk_size, dim=0)
        #     assert len(context_layer_splits) == len(residual_splits), \
        #         f"len: {len(context_layer_splits)} - {len(residual_splits)}, context_layer shape: {context_layer.shape}, residual shape: {residual.shape}"
        #     o = []
        #     for context_layer_split, residual_split in zip(context_layer_splits, residual_splits):
        #         o.append(self.single_forward(context_layer_split, residual_split))
        #     o = torch.cat(o, dim=0)
        # else:
        o = self.single_forward(context_layer, residual)
        
        output = core.utils.make_viewless_tensor(inp = o,
                                                requires_grad = o.requires_grad,
                                                keep_graph = True)
        return output

    def single_forward(self, context_layer, residual):
        ############### kernel 4 ################
        attention_output, attention_bias = self.dense(context_layer)
        
        assert self.drop_path is None, "Do not support drop path for now"

        # jit scripting for a nn.module (with dropout) is not
        # trigerring the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        if attention_bias is not None:
            attention_bias = attention_bias.expand_as(residual)
        with self.bias_dropout_add_exec_handler():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias,
                residual,
                self.hidden_dropout)

        ############### kernel 5 ################
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.chunk_size > 0:
            mlp_output, mlp_bias = self.chunked_mlp(layernorm_output, self.chunk_size)
        else:
            mlp_output, mlp_bias = self.full_mlp(layernorm_output)
        assert mlp_bias is None

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if mlp_bias is not None:
            mlp_bias = mlp_bias.expand_as(residual)
        with self.bias_dropout_add_exec_handler():
            output = bias_dropout_add_func(
                mlp_output,
                mlp_bias,
                residual,
                self.hidden_dropout)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        # output = core.utils.make_viewless_tensor(inp = output,
        #                                         requires_grad = output.requires_grad,
        #                                         keep_graph = True)

        return output

    def full_mlp(self, layernorm_output):
        ############### kernel 6 ################
        # MLP - Linear 1
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(layernorm_output)

        ############### kernel 7 ################
        if self.bias_gelu_fusion:
            assert self.add_bias is True
            assert self.activation_func == F.gelu
            if self.use_triton_fusion:
                activated_parallel = fused_bias_gelu_triton(intermediate_parallel, bias_parallel)

            else:
                activated_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            activated_parallel = self.activation_func(intermediate_parallel)

        ############### kernel 8 ################
        # [s, b, h]
        mlp_output, mlp_bias = self.dense_4h_to_h(activated_parallel)

        return mlp_output, mlp_bias

    def chunked_mlp(self, layernorm_output, chunk_size):
        assert self.bias_gelu_fusion and self.use_triton_fusion, "only support bias_gelu_fusion with triton fusion"
        mlp_output = ChunkedMLP.apply(
            layernorm_output, self.dense_h_to_4h.weight, self.dense_h_to_4h.bias,
            self.dense_4h_to_h.weight, self.dense_4h_to_h.bias, chunk_size, self.sequence_parallel)
        return mlp_output, None


class PreFlashAttnCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, dropout_p, training, *args):
        # hard coded options for newer flash attn kernel
        # window_size=(-1, -1)
        # alibi_slopes=None
        return_attn_probs = False
        ctx.run_function = run_function
        
        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        with torch.no_grad():
            q, k, v, batch_size, max_seqlen_q, max_seqlen_k = run_function(*args)
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * max_seqlen_q, step=max_seqlen_q, dtype=torch.int32,
                                    device=q.device)
            if training:
                causal = True
                assert max_seqlen_q == max_seqlen_k
                cu_seqlens_k = cu_seqlens_q
            else:
                causal = max_seqlen_q == max_seqlen_k
                cu_seqlens_k = torch.arange(0, (batch_size + 1) * max_seqlen_k, step=max_seqlen_k, dtype=torch.int32,
                        device=q.device)
                dropout_p = 0
            
            softmax_scale = q.shape[-1] ** (-0.5)
            
            assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
            # To ensure q, k, v can be discarded safely, check if their head size can be diviede by 8
            # ref: https://github.com/Dao-AILab/flash-attention/blob/9356a1c0389660d7e231ff3163c1ac17d9e3824a/csrc/flash_attn/flash_api.cpp#L511
            assert q.shape[2]%8==0, f"q head size must be divisible by 8, current size: {q.shape[2]}"
            assert k.shape[2]%8==0, f"k head size must be divisible by 8, current size: {k.shape[2]}"
            assert v.shape[2]%8==0, f"v head size must be divisible by 8, current size: {v.shape[2]}"

            get_pipeline_context().offload()
            
            out, _, _, _, out_padded, softmax_lse, _, rng_state = _flash_attn_varlen_forward(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
            softmax_scale, causal, 
            # window_size, alibi_slopes, 
            return_attn_probs)

        # To ensure we can replace `outpadded` with `out`, check if its head size can be divided by 8
        # ref: https://github.com/Dao-AILab/flash-attention/blob/9356a1c0389660d7e231ff3163c1ac17d9e3824a/csrc/flash_attn/flash_api.cpp#L598
        assert out is out_padded, f"output head size must be divisible by 8, current size: {out.shape[2]}"

        # Store everything.
        ctx.save_for_backward(out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state, *args)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        # ctx.window_size = window_size
        # ctx.alibi_slopes = alibi_slopes
        return out

    @staticmethod
    def backward(ctx, dout):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state, *inputs = ctx.saved_tensors
        
        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(tuple(inputs))
        with torch.enable_grad():
            q, k, v, _, _, _ = ctx.run_function(*detached_inputs)
        
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        
        get_pipeline_context().upload()
        _flash_attn_varlen_backward(
            dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k, 
            ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.dropout_p, ctx.softmax_scale,
            ctx.causal, 
            # ctx.window_size, ctx.alibi_slopes, deterministic=False,
            rng_state=rng_state,)

        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)
        
        torch.autograd.backward((q, k, v), (dq, dk, dv))

        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None, None) + grads


def checkpoint_pre_flash_attn(prefunc, dropout_p=0.0, training=True, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return PreFlashAttnCheckpointFunction.apply(prefunc, dropout_p, training, *args)


class AttentionOp:
    def __init__(self, args):
        super(AttentionOp, self).__init__()
        
        tp_size = mpu.get_tensor_model_parallel_world_size()
        self.transfer_weight = args.transfer_weight
        self.checkpoint_without_attn = args.checkpoint_without_attn

        self.attn_context = nullcontext if args.sequence_parallel \
            else tensor_parallel.get_cuda_rng_tracker().fork 
        
        if self.transfer_weight:
            self.async_tensor_model_parallel_allreduce = (
                args.async_tensor_model_parallel_allreduce and tp_size > 1
            )

            self.sequence_parallel = args.sequence_parallel
            if self.sequence_parallel and tp_size <= 1:
                self.sequence_parallel = False
            # This has to be False
            self.gradient_accumulation_fusion = False #args.gradient_accumulation_fusion

        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, tp_size)
    
        if args.group_query_attention:
            self.num_query_groups_per_partition = core.utils.divide(
                args.num_query_groups, tp_size)
        else:
            self.num_query_groups_per_partition = self.num_attention_heads_per_partition
        
        query_projection_size = args.kv_channels * args.num_attention_heads
        self.hidden_size_per_attention_head = core.utils.divide(
            query_projection_size, args.num_attention_heads)
        self.dropout_p = args.attention_dropout
    
    def pre_func(self, *inp_activation):
        if self.transfer_weight:
            assert isinstance(inp_activation, (tuple, list)) and len(inp_activation) == 3, \
                f"type: {type(inp_activation)}, len: {len(inp_activation)}"
            
            layernorm_output, weight, bias = inp_activation
            # Self attention.
            ############### kernel 2 ###############
            # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
            # here, np = ng
            mixed_x_layer = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
                input=layernorm_output,
                weight=weight,
                bias=bias,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
                sequence_parallel=self.sequence_parallel,)

        else:
            assert len(inp_activation) == 1, f"len: {len(inp_activation)}"
            (mixed_x_layer, ) = inp_activation
        
        # x --> q, k, v
        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_query_groups_per_partition,
                (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                    * self.hidden_size_per_attention_head
                ),
        )
        new_mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query_layer,
        key_layer,
        value_layer) = torch.split(
            new_mixed_x_layer,
            [
                (
                    self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head
            ],
            dim=3)
        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] -
        query_layer = query_layer.view(query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head)
        
        key_layer = key_layer.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
            dim = 2
        )
        value_layer = value_layer.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
            dim = 2
        )
        
        ############### prepare for kernel 3 ################
        batch_size, seqlen_q = query_layer.shape[1], query_layer.shape[0]
        seqlen_k = key_layer.shape[0]

        q, k, v = [rearrange(x, 's b ... -> (b s) ...').contiguous()
                    for x in (query_layer, key_layer, value_layer)]
        
        return q, k, v, batch_size, seqlen_q, seqlen_k

    def __call__(self, inp_activation: Tuple[torch.Tensor]):
        if self.checkpoint_without_attn:
            with self.attn_context():
                context_layer = checkpoint_pre_flash_attn(self.pre_func, self.dropout_p, True, *inp_activation)
            
            # get_pipeline_context().push_imbo([*inp_activation, context_layer,])
            # inp_activation[0].register_hook(lambda grad: get_pipeline_context().pop_attn_act())
        else:
            q, k, v, batch_size, seqlen_q, seqlen_k = self.pre_func(*inp_activation)

            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                        device=q.device)
            
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q
            is_causal = True
            cu_seqlens_k = cu_seqlens_q
            with self.attn_context():
                context_layer = flash_attn_varlen_func(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
                    self.dropout_p, softmax_scale=None, causal=is_causal
                )
        return context_layer


class CustomizedLayer(MegatronModule):
    """
    This class is for testing the sanity of PreAttn, AttentionOp and PostAttn

    the unit test is in tests/unit_test_layer.py
    """
    def __init__(self, config,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
        args = get_args()
        assert drop_path_rate == 0.

        super(CustomizedLayer, self).__init__()
        self.layer_number = layer_number
        self.checkpoint_without_attn = args.checkpoint_without_attn
        self.bf16 = config.bf16

        self.pre_attn = PreAttnModule(config, args, layer_number, self_attn_mask_type)
        self.attn = AttentionOp(args)
        self.post_attn = PostAttnModule(config, args, layer_number)

    def forward(self, hidden_states, attention_mask):
        if self.checkpoint_without_attn:
            *mixed_x_layer, residual = tensor_parallel.checkpoint(self.pre_attn, False, hidden_states)
        else:
            *mixed_x_layer, residual = self.pre_attn(hidden_states)

        context_layer = self.attn(mixed_x_layer)

        if self.checkpoint_without_attn:
            output = tensor_parallel.checkpoint(self.post_attn, False, context_layer, residual)
        else:
            output = self.post_attn(context_layer, residual)

        return output


class CustomizedGPT(MegatronModule):

    def __init__(self, config, args, layer_idx, pre_process=False, post_process=False):
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.residual_tensor = None

        if pre_process:
            self.embedding = Embedding(
                config.hidden_size, args.padded_vocab_size,
                args.max_position_embeddings, args.hidden_dropout,
                config, 0, args.embedding_weights_in_fp32)
            modules = [PreAttnModule(config, args, layer_idx+1, AttnMaskType.causal),]
            # for debug purpose
            self.component_key = [f"pre_{layer_idx+1}",]
        elif post_process:
            assert layer_idx == config.num_layers - 1
            modules = [
                PostAttnModule(config, args, config.num_layers),
                # output layernorm
                LayerNorm(
                    config.hidden_size, eps=config.layernorm_epsilon,
                    no_persist_layer_norm=args.no_persist_layer_norm,
                    sequence_parallel=config.sequence_parallel,
                    apply_layernorm_1p=args.apply_layernorm_1p)
            ]
            self.component_key = [f"post_{config.num_layers}"]
        else:
            modules = [
                PostAttnModule(config, args, layer_idx, ),
                PreAttnModule(config, args, layer_idx+1, AttnMaskType.causal)
            ]
            self.component_key = [f"post_{layer_idx}", f"pre_{layer_idx+1}"]
        
        self.components = torch.nn.ModuleList(modules)

        self.checkpoint_without_attn = args.checkpoint_without_attn

        self.parallel_output = True
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def set_residual_tensor(self, residual_tensor):
        self.residual_tensor = residual_tensor

    def _get_post_pre_func(self):
        def post_pre_func(context_layer, residual):
            o = self.components[0](context_layer, residual)
            # if post process: hidden states
            # else: (qkv, residual)
            return self.components[1](o)
        return post_pre_func

    def forward(self, input_ids, position_ids):
        # return *qkv, residual
        # pipeline_ctx = get_pipeline_context()

        if self.pre_process:
            hidden_states = self.embedding(input_ids, position_ids,)
            if self.checkpoint_without_attn:
                # pipeline_ctx.push_start([hidden_states,])
                # hidden_states.register_hook(lambda grad: pipeline_ctx.pop_start_act())

                outputs = tensor_parallel.random.checkpoint(self.components[0], False, hidden_states)
                return outputs
            else:
                return self.components[0](hidden_states)
        
        # return [*qkv, residual] or hidden_states before loss
        if self.checkpoint_without_attn:
            input_tensor = self.input_tensor
            residual_tensor = self.residual_tensor

            # input_tensor.register_hook(lambda grad: pipeline_ctx.pop_param_act())

            def post_pre_func(o, r):
                return self.components[1](self.components[0](o, r))
            output_states = tensor_parallel.random.checkpoint(
                post_pre_func, False, input_tensor, residual_tensor)
            # if self.post_process:
            #     pipeline_ctx.push_end([residual_tensor, input_tensor, output_states,])
            # else:
            #     pipeline_ctx.push_ro([residual_tensor, input_tensor])

        else:
            output_states =  self.components[1](self.components[0](self.input_tensor, self.residual_tensor))

        self.input_tensor = None
        self.residual_tensor = None
        return output_states


def attention_pipeline_model_provider():
    args = get_args()
    config = core_transformer_config_from_args(args)
    pipeline_size = mpu.get_pipeline_model_parallel_world_size()
    pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    num_loop = mpu.get_virtual_pipeline_model_parallel_world_size()
    model = []
    assert config.num_layers % pipeline_size == 0
    
    for i in range(num_loop):
        mpu.set_virtual_pipeline_model_parallel_rank(i)
        layer_idx = i*pipeline_size + pipeline_rank
        if layer_idx == 0:
            model.append(CustomizedGPT(config, args, layer_idx, pre_process=True, post_process=False))
        else:
            model.append(CustomizedGPT(config, args, layer_idx, pre_process=False, post_process=False))
    if pipeline_rank == 0:
        model.append(CustomizedGPT(config, args, config.num_layers-1, pre_process=False, post_process=True))
    return model


def get_batch(data_iterator):
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    _, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        args.no_attention_mask)
    timers('batch-generator').stop()

    return tokens, labels, loss_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def attention_pipeline_forward_step(
    tokens, position_ids, model
):
    return model(tokens, position_ids)


def get_forward_step_func(args):
    attn_func = AttentionOp(args)
    
    return (attn_func, attention_pipeline_forward_step)
