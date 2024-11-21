import torch

import triton
import triton.language as tl


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
#     ],
#     key=['n_elements'],
# )
@triton.jit
def emb_add_dropout_fwd_kernel(
    emb_ptr,
    pos_emb_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    emb = tl.load(emb_ptr+offsets, mask=mask)
    pos_emb = tl.load(pos_emb_ptr+offsets, mask=mask)
    x = emb + pos_emb

    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
#     ],
#     key=['n_elements'],
# )
@triton.jit
def emb_add_dropout_bwd_kernel(
    grad_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from grad ptr
    mask = offsets < n_elements
    grad = tl.load(grad_ptr+offsets, mask=mask)

    # rematerialize dropout mask
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, grad / (1-p), 0.)
    tl.store(output_ptr+offsets, output, mask=mask)


class EmbeddingAddDropout(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, emb, pos_emb, dropout_prob):
        output = torch.empty_like(emb)
        assert emb.is_contiguous() and pos_emb.is_contiguous()

        seed = torch.randint(low=0, high=torch.iinfo(torch.long).max, size=(1,), dtype=torch.long).item()

        emb_args = emb.view(-1)
        pos_emb_args = pos_emb.view(-1)
        output_args = output.view(-1)
        n_elements = emb_args.shape[0]
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        emb_add_dropout_fwd_kernel[grid](
            emb_args, pos_emb_args, output_args, n_elements, dropout_prob,
            seed, BLOCK_SIZE=1024)

        ctx.dropout_prob = dropout_prob
        ctx.seed = seed
        return output

    @staticmethod
    def backward(ctx, grad):
        input_grad = torch.empty_like(grad)
        
        grad_args = grad.view(-1)
        input_grad_args = input_grad.view(-1)
        n_elements = grad_args.shape[0]
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        emb_add_dropout_bwd_kernel[grid](
            grad_args, input_grad_args, n_elements,
            ctx.dropout_prob, ctx.seed, BLOCK_SIZE=1024)

        return input_grad, input_grad, None


embedding_add_dropout = EmbeddingAddDropout.apply
