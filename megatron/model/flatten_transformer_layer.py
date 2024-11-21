from typing import Optional
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import detach_variable

from einops import rearrange
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, \
    _flash_attn_varlen_backward, flash_attn_varlen_func

from megatron import get_args, core
from megatron.core import tensor_parallel, parallel_state as mpu
from megatron.model import LayerNorm
from megatron.model.module import MegatronModule
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.triton_bias_gelu import fused_bias_gelu_triton
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker,\
    _set_cuda_rng_state, checkpoint
from megatron.core.pipeline_parallel import get_activation_queue

# duplicated jit functions will cause rng errors!!!
# @torch.jit.script
# def bias_dropout_add_fused_train(x: torch.Tensor,
#                                  bias: Optional[torch.Tensor],
#                                  residual: torch.Tensor,
#                                  prob: float) -> torch.Tensor:
#     return bias_dropout_add(x, bias, residual, prob, True)


class PreFlashAttnCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, dropout_p, causal, training, offload_activation, *args):
        # hard coded options for flash attn kernel
        # window_size=(-1, -1) # not in v2.0.4
        return_attn_probs = False

        ctx.run_function = run_function

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        act_queue = None
        if offload_activation:
            act_queue = get_activation_queue()
        with torch.no_grad():
            q, k, v, residual, batch_size, max_seqlen_q, max_seqlen_k = run_function(*args)

            cu_seqlens_q = torch.arange(0, (batch_size + 1) * max_seqlen_q, step=max_seqlen_q, dtype=torch.int32,
                                    device=q.device)
            if training:
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

            if act_queue is not None:
                act_queue.forward()
            # out, padded_q, padded_k, padded_v, out_padded, softmax_lse, _, rng_state = _flash_attn_varlen_forward(
            out, _, _, _, out_padded, softmax_lse, _, rng_state = _flash_attn_varlen_forward(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
            softmax_scale, causal, return_attn_probs)

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
        ctx.offload_activation = offload_activation
        # ctx.window_size = window_size
        return out, residual

    @staticmethod
    def backward(ctx, dout, dresidual):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state, *inputs = ctx.saved_tensors
        act_queue = get_activation_queue() if ctx.offload_activation else None

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
            q, k, v, residual, _, _, _ = ctx.run_function(*detached_inputs)

        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        if act_queue is not None:
            act_queue.backward()
        _flash_attn_varlen_backward(
            dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k, 
            ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.dropout_p, ctx.softmax_scale,
            ctx.causal, 
            # ctx.window_size, 
            rng_state=rng_state,)

        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)
        
        torch.autograd.backward((q, k, v, residual), (dq, dk, dv, dresidual))

        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None, None, None, None) + grads


def checkpoint_pre_flash_attn(prefunc, dropout_p=0.0,
    causal=False, training=True, offload_activation=False, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return PreFlashAttnCheckpointFunction.apply(prefunc, dropout_p, causal, training, offload_activation, *args)


class FlattenTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
        args = get_args()

        super(FlattenTransformerLayer, self).__init__()
        self.checkpoint_without_attn = args.checkpoint_without_attn
        self.offload_activation = args.offload_activation
        self.act_queue = get_activation_queue() if self.offload_activation else None

        self.layer_number = max(1, layer_number)
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = config.apply_residual_connection_post_layernorm

        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        
        ############### Attention BLOCK ################
        ############### kernel 1 ################
        # Layernorm 1 on the input data.
        self.input_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=config.sequence_parallel,
            apply_layernorm_1p=args.apply_layernorm_1p)

        ############### kernel 2 ################
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
        
        ############### kernel 3 ################
        # Self attention - flash attn
        self.dropout_p = config.attention_dropout

        ############### kernel 4 ################
        # Self attention - O Linear
        self.dense = tensor_parallel.RowParallelLinear(
            query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=args.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True)
        # hidden dropout is appended to O Linear
        self.hidden_dropout = config.hidden_dropout
        self.bias_dropout_fusion = config.bias_dropout_fusion
        assert drop_path_rate == 0, f"do not support drop path"
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

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
                nullcontext if use_nvfuser else torch.enable_grad
        self.retriever = None

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

    def _add_mlp_block_attrs(self, args, config):
        self.add_bias = config.add_bias_linear

        assert config.ffn_hidden_size // 4 == config.hidden_size, \
            f"only support gelu"
        
        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        self.swiglu = args.swiglu

    def _get_pre_attn_func(self):
        def pre_attn_func(hidden_states):
            ############### kernel 1 ################
            # Layer norm at the beginning of the transformer layer.
            layernorm_output = self.input_layernorm(hidden_states)

            # Self attention.
            ############### kernel 2 ################
            # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
            mixed_x_layer, _ = self.query_key_value(layernorm_output)

            # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                    self.num_query_groups_per_partition,
                    (
                        (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                        * self.hidden_size_per_attention_head
                    ),
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query_layer,
            key_layer,
            value_layer) = torch.split(
                mixed_x_layer,
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

            batch_size, seqlen_q = query_layer.shape[1], query_layer.shape[0]
            seqlen_k = key_layer.shape[0]

            q, k, v = [rearrange(x, 's b ... -> (b s) ...').contiguous()
                        for x in (query_layer, key_layer, value_layer)]
            
            # prepare for kernel 4
            # Residual connection.
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = hidden_states
            return q, k, v, residual, batch_size, seqlen_q, seqlen_k
        return pre_attn_func
    
    def _get_post_attn_func(self):
        def post_attn_func(attn_output, residual):
            batch_size = residual.shape[1]
            # output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            context_layer = rearrange(attn_output, '(b s) h d -> s b (h d)', b=batch_size).contiguous()

            ############### kernel 4 ################
            attention_output, attention_bias = self.dense(context_layer)

            if self.drop_path is None:
                # jit scripting for a nn.module (with dropout) is not
                # trigerring the fusion kernel. For now, we use two
                # different nn.functional routines to account for varying
                # dropout semantics during training and inference phases.
                from megatron.model.transformer import get_bias_dropout_add, \
                    bias_dropout_add_fused_train, bias_dropout_add_fused_inference
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
            else:
                raise KeyError("do not support drop path")

            ############### kernel 5 ################
            # Layer norm post the self attention.
            layernorm_output = self.post_attention_layernorm(layernorm_input)

            ############### kernel 6 ################
            # MLP - Linear 1
            intermediate_parallel, bias_parallel = self.dense_h_to_4h(layernorm_output)
            
            ############### kernel 7 ################
            if self.bias_gelu_fusion:
                assert self.add_bias is True
                assert self.activation_func == F.gelu
                if self.use_triton_fusion:
                    intermediate_parallel = fused_bias_gelu_triton(intermediate_parallel, bias_parallel)
                else:
                    intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
            else:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                intermediate_parallel = self.activation_func(intermediate_parallel)

            ############### kernel 8 ################
            # [s, b, h]
            mlp_output, mlp_bias = self.dense_4h_to_h(intermediate_parallel)
            
            # Second residual connection.
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            if self.drop_path is None:
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
                output = core.utils.make_viewless_tensor(inp = output,
                                                        requires_grad = output.requires_grad,
                                                        keep_graph = True)
            else:
                raise KeyError("Do not support drop path")

            return output

        return post_attn_func

    def checkpointed_forward(self, hidden_states):
        attn_output, residual = checkpoint_pre_flash_attn(
            self._get_pre_attn_func(), self.dropout_p, 
            True, self.training, self.offload_activation, hidden_states)
        
        output = checkpoint(self._get_post_attn_func(), False, attn_output, residual)
        if self.act_queue is not None:
            self.act_queue.push((hidden_states, attn_output))
            def hook(grad):
                self.act_queue.pop()
            output.register_hook(hook)
        return output
    
    def straight_forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None):
        """
        archive only
        """
        # hidden_states: [sq, b, h]

        ############### kernel 1 ################
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        assert inference_params is None
        assert rotary_pos_emb is None
        ############### kernel 2 ################
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_x_layer, _ = self.query_key_value(layernorm_output)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_query_groups_per_partition,
                (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                    * self.hidden_size_per_attention_head
                ),
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query_layer,
        key_layer,
        value_layer) = torch.split(
            mixed_x_layer,
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

        batch_size, seqlen_q = query_layer.shape[1], query_layer.shape[0]
        seqlen_k = key_layer.shape[0]

        q, k, v = [rearrange(x, 's b ... -> (b s) ...').contiguous()
                       for x in (query_layer, key_layer, value_layer)]

        # q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
        
        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = True
            cu_seqlens_k = cu_seqlens_q
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=q.device)
            dropout_p = 0

        ############### kernel 3 ################
        assert self.sequence_parallel, "must enable sequence parallel"
        output = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            dropout_p, causal=is_causal)
        
        # output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        context_layer = rearrange(output, '(b s) h d -> s b (h d)', b=batch_size).contiguous()

        ############### kernel 4 ################
        attention_output, attention_bias = self.dense(context_layer)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.drop_path is None:
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
        else:
            raise KeyError("do not support drop path")

        ############### kernel 5 ################
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        ############### kernel 6 ################
        # MLP - Linear 1
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(layernorm_output)
        
        ############### kernel 7 ################
        if self.bias_gelu_fusion:
            assert self.add_bias is True
            assert self.activation_func == F.gelu
            if self.use_triton_fusion:
                intermediate_parallel = fused_bias_gelu_triton(intermediate_parallel, bias_parallel)
            else:
                intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)

        ############### kernel 8 ################
        # [s, b, h]
        mlp_output, mlp_bias = self.dense_4h_to_h(intermediate_parallel)
        
        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
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
            output = core.utils.make_viewless_tensor(inp = output,
                                                     requires_grad = output.requires_grad,
                                                     keep_graph = True)
        else:
            raise KeyError("Do not support drop path")

        return output

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None):
        if self.checkpoint_without_attn:
            assert mpu.get_tensor_model_parallel_world_size() == 1 or self.sequence_parallel, "sequence parallel is required"
            return self.checkpointed_forward(hidden_states)
        else:
            # return self.straight_forward(
            #     hidden_states, attention_mask,
            #     encoder_output, enc_dec_attn_mask,
            #     retriever_input, retriever_output,
            #     retriever_attn_mask, inference_params,
            #     rotary_pos_emb)

            q, k, v, residual, batch_size, max_seqlen_q, max_seqlen_k = self._get_pre_attn_func()(hidden_states)
            
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * max_seqlen_q,
                                        step=max_seqlen_q, dtype=torch.int32,
                                        device=q.device)
            causal = True
            dropout_p = self.dropout_p
            if self.training:
                assert max_seqlen_q == max_seqlen_k
                cu_seqlens_k = cu_seqlens_q
            else:
                causal = max_seqlen_q == max_seqlen_k
                cu_seqlens_k = torch.arange(0, (batch_size + 1) * max_seqlen_k, 
                                            step=max_seqlen_k, dtype=torch.int32,
                                            device=q.device)
                dropout_p = 0
            context = nullcontext if self.sequence_parallel else tensor_parallel.get_cuda_rng_tracker().fork
            with context():
                attn_output = flash_attn_varlen_func(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
                    dropout_p, causal=causal)

            return self._get_post_attn_func()(attn_output, residual)
