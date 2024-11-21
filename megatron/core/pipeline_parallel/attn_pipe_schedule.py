from typing import Iterator, List, Union
from contextlib import nullcontext

import torch
from torch.autograd.variable import Variable

from megatron import get_args
from megatron.core import parallel_state
from megatron.core.utils import get_model_config, get_attr_wrapped_model
from megatron.model.gpt_model import post_language_model_processing
from megatron.model.customized_modules import get_batch, loss_func
from .pipeline_context import get_pipeline_context


def get_tensor_shapes(args):
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    activation_shape = [args.seq_length// tp_size, args.micro_batch_size, args.hidden_size]
    o_shape = [
        args.seq_length * args.micro_batch_size, 
        args.num_attention_heads//tp_size, 
        args.hidden_size//args.num_attention_heads]
    
    if args.transfer_weight:
        pre_shape = [
            [args.seq_length//tp_size, args.micro_batch_size, args.hidden_size], # qkv input
            [args.hidden_size*3//tp_size, args.hidden_size],  # qkv linear weight
            [args.hidden_size*3//tp_size,]]  # qkv linear bias
    else:
        pre_shape = [[args.seq_length, args.micro_batch_size, args.hidden_size//tp_size*3]]
    return pre_shape, o_shape, activation_shape


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    loop_idx,
    residual_attn_output,
    config,
):
    residual, attn_output = residual_attn_output
    cur_data_iterator = data_iterator[loop_idx]
    cur_model = model[loop_idx]
    set_input_tensor = get_attr_wrapped_model(cur_model, "set_input_tensor")
    set_residual_tensor = get_attr_wrapped_model(cur_model, "set_residual_tensor")
    set_input_tensor(attn_output)
    set_residual_tensor(residual)
    
    if get_attr_wrapped_model(cur_model, "pre_process"):
        tokens, labels, loss_mask, position_ids = get_batch(cur_data_iterator)
        get_pipeline_context().register_labels((labels, loss_mask))
    else:
        tokens, position_ids = None, None
    
    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = nullcontext()
    with context_manager:
        output_tensor = forward_step_func(tokens, position_ids, cur_model)

    return output_tensor

# """
def backward_step(
        inp_tensors, oup_tensors, oup_grad, config, 
        preprocess_model=None, fp16_lm_cross_entropy=False, num_microbatches=None, 
        forward_data_store=None, collect_non_loss_data=False):
    # loss scale
    if oup_grad[0] is None:
        assert len(oup_tensors) == 1
        assert get_attr_wrapped_model(preprocess_model, "pre_process")

        labels, loss_mask = get_pipeline_context().pop_labels()
        embedding = get_attr_wrapped_model(preprocess_model, "embedding")
        embed_matrix = embedding.word_embeddings.weight
        
        loss = post_language_model_processing(
            oup_tensors[0], labels, embed_matrix,
            parallel_output=True, fp16_lm_cross_entropy=fp16_lm_cross_entropy)

        if not collect_non_loss_data:
            loss, loss_reduced = loss_func(loss_mask, loss)
            output_tensor = loss / num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(loss_mask, loss, non_loss_data=True)
            forward_data_store.append(data)

        oup_tensors[0] = output_tensor
        if config.grad_scale_func is not None:
            oup_tensors[0] = config.grad_scale_func(oup_tensors[0])
        oup_grad[0] = torch.ones_like(oup_tensors[0])

    # Backward pass.
    # Copied from megatron, omit shape check
    Variable._execution_engine.run_backward(
        tensors=tuple(oup_tensors),
        grad_tensors=tuple(oup_grad),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )

    # Collect the grad of the input_tensor.
    input_tensor_grad = []
    for x in inp_tensors:
        if x is None or not x.is_leaf:
            input_tensor_grad.append(None)
        else:
            input_tensor_grad.append(x.grad)
    
    return input_tensor_grad
# """

"""
def backward_step(inp_tensors, oup_tensors, oup_grad, config):
    '''
    dummy backward pass for debug purpose only

    2, 2: (ro, ro), (ro, qkvr), (qkvr, qkvr)
    2, 1: (ro, loss), (qkvr, loss)
    1, 1: (qkv, o)
    '''
    # (ro, loss), (qkvr, loss)
    if (len(inp_tensors) == 2 or len(inp_tensors)==4) \
        and len(oup_tensors) == 1:
        assert oup_grad[0] is None
        num_microbatches = get_num_microbatches()
        grad_val = oup_tensors[0].view(-1)[0].item() * num_microbatches - 1
        
        lval = inp_tensors[0].view(-1)[0].item()
        rval = inp_tensors[-1].view(-1)[0].item()

        if lval > rval:  # (qkvr, loss)
            *qkv, r = inp_tensors

            for each in qkv:
                inp_grad = torch.ones_like(each) * grad_val
                assert torch.allclose(each, inp_grad), f"qkv: {each}\nqkv_grad: {inp_grad}, loss: {oup_tensors[0]}"
                each.grad = inp_grad

            r_grad = torch.ones_like(r) * (grad_val - 1)
            assert torch.allclose(r, r_grad), f"r: {r}\nr_grad: {r_grad}"
            r.grad = r_grad
        else:  # (ro, loss)
            r, o = inp_tensors

            r_grad = torch.ones_like(r) * (grad_val - 1)
            assert torch.allclose(r, r_grad), f"r: {r}\nr_grad: {r_grad}, loss: {oup_tensors[0]}"
            r.grad = r_grad

            o_grad = torch.ones_like(o) * grad_val
            assert torch.allclose(o, o_grad), f"o: {o}\no_grad: {o_grad}"
            o.grad = o_grad
    
    # (qkv, o)
    elif (len(inp_tensors) == 1 or len(inp_tensors)==3) \
        and len(oup_tensors) == 1:
        qkv = inp_tensors
        grad_val = oup_grad[0].view(-1)[0].item()
        
        for each in qkv:
            inp_grad = torch.ones_like(each) * grad_val
            assert torch.allclose(each, inp_grad), f"qkv: {each}\nqkv_grad: {inp_grad}, loss: {oup_tensors[0]}"
            each.grad = inp_grad
    
    # (ro, ro), (ro, qkvr), (qkvr, qkvr)
    elif (len(inp_tensors) == 2 or len(inp_tensors)==4) \
        and (len(oup_tensors) == 2 or len(oup_tensors)==4):
        if inp_tensors[0] is None:
            loup = oup_tensors[0].view(-1)[0].item()
            roup = oup_tensors[-1].view(-1)[0].item()
            if loup > roup: # qkv, r
                grad_val = oup_grad[-1].view(-1)[0].item()
            else: # r o
                grad_val = oup_grad[0].view(-1)[0].item()
            assert grad_val == config, f"r: {grad_val}, oup grad 0: {oup_grad[0]}, oup grad 1: {oup_grad[-1]} target: {config}"
        else:
            lval = inp_tensors[0].view(-1)[0].item()
            rval = inp_tensors[-1].view(-1)[0].item()
            if lval > rval:  # (qkvr, qkvr)
                *qkv, r = inp_tensors
                grad_val = oup_grad[-1].view(-1)[0].item()

                for each in qkv:
                    inp_grad = torch.ones_like(each) * grad_val
                    assert torch.allclose(each, inp_grad), f"qkv: {each}\nqkv_grad: {inp_grad}, loss: {oup_tensors[0]}"
                    each.grad = inp_grad

                r_grad = torch.ones_like(r) * (grad_val-1)
                assert torch.allclose(r_grad, r), f"r: {r}\nr_grad: {r_grad}"
                r.grad = r_grad
            elif lval < rval: # (ro, ro), (ro, qkvr)
                r, o = inp_tensors
                loup = oup_tensors[0].view(-1)[0].item()
                roup = oup_tensors[-1].view(-1)[0].item()
                
                if loup > roup:  # (ro, qkvr)
                    grad_val = oup_grad[-1].view(-1)[0].item()
                else:  # ro, ro
                    grad_val = oup_grad[0].view(-1)[0].item()

                o_grad = torch.ones_like(o) * grad_val
                assert torch.allclose(o_grad, o), \
                    f"rank: {parallel_state.get_pipeline_model_parallel_rank()} o: {o}\no_grad: {o_grad}, {len(oup_grad)}"
                o.grad = o_grad

                r_grad = torch.ones_like(r) * (grad_val-1)
                assert torch.allclose(r_grad, r), f"r: {r}\nr_grad: {r_grad}"
                r.grad = r_grad
            else:
                raise ValueError(f"pp rank {parallel_state.get_pipeline_model_parallel_rank()}: lval: {lval}, rval: {rval}")
    else:
        raise ValueError(f"len inp: {len(inp_tensors)}, oup: {len(oup_tensors)}")
    
    inp_grad = []
    for each in inp_tensors:
        inp_grad.append(each.grad if each is not None else None)
    return inp_grad
"""

def attention_pipeline(
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
):
    """
    multi-fold attention pipeline
    
    Comments: 
    - data offset: the batch idx in a data repeat  
    - rank offset: the src rank for a batch
    """
    args = get_args()
    pipeline_context = get_pipeline_context()
    pipeline_context.reset()
    config = get_model_config(model[0])

    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    num_loop = parallel_state.get_virtual_pipeline_model_parallel_world_size()

    num_fold = args.num_fold
    assert num_microbatches % pp_size == 0 and num_microbatches % num_fold == 0
    num_data_repeat = num_microbatches // pp_size // num_fold
    
    attn_func, forward_step_func = forward_step_func

    pre_shapes, o_shape, r_shape = get_tensor_shapes(args)
    forward_data_store = []
    
    # control mixed precision module
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    for fold in range(num_fold):
        if pp_rank == 0:
            pipeline_context.recv_ro(
                data_repeat_idx=0, loop_idx=0, data_offset=0, fold_idx=fold,
                r_shape=None, o_shape=None, src_tgt=None, config=config)
            
            r, o = pipeline_context.get_ro(
                data_repeat_idx=0, loop_idx=0, data_offset=0, fold_idx=fold)
            
            *qkv, r = forward_step(
                forward_step_func, data_iterator, model, 0, (r, o), config)

            src_tgt = pp_rank+1

            # data offset + 1
            pipeline_context.recv_ro_send_qkvr(
                data_repeat_idx=0, loop_idx=0, data_offset=1, fold_idx=fold,
                r_shape=None, o_shape=None, qkv_tensor=qkv, r_tensor=r, src_tgt=src_tgt, config=config)
            
            pipeline_context.deallocate_output_tensor(qkv[0])

            pipeline_context.register_qkvr(
                data_repeat_idx=0, loop_idx=0, data_offset=0, fold_idx=fold,
                qkv=qkv, r=r)
            
            del qkv, r, o
        else:
            pipeline_context.recv_qkvr(
                data_repeat_idx=0, loop_idx=0, data_offset=pp_rank-1, fold_idx=fold,
                qkv_shape=pre_shapes, r_shape=r_shape, src_tgt=0, config=config)

    for repeat_idx in range(num_data_repeat):
        # warm up
        # print("="*25 + f" rank {pp_rank} repeat {repeat_idx} warm up " + "="*25)
        parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        # attn part
        for rank_offset, data_offset in enumerate(reversed(range(pp_rank))):
            for fold in range(num_fold):
                qkv, r = pipeline_context.get_qkvr(
                    data_repeat_idx=repeat_idx, loop_idx=0, data_offset=data_offset, fold_idx=fold)
                
                o = attn_func(qkv)
                
                # transition from attn to param
                if data_offset == 0:
                    *qkv, r = forward_step(
                        forward_step_func, data_iterator, model, 0, (r, o), config)

                    src_tgt = (pp_rank+data_offset+1)%pp_size
                    pipeline_context.recv_ro_send_qkvr(
                        data_repeat_idx=repeat_idx, loop_idx=0, data_offset=data_offset+1, fold_idx=fold,
                        r_shape=r_shape, o_shape=o_shape, qkv_tensor=qkv, r_tensor=r, 
                        src_tgt=src_tgt, config=config)
                    
                    pipeline_context.deallocate_output_tensor(qkv[0])
                    if args.checkpoint_without_attn:
                        pipeline_context.deallocate_output_tensor(r)

                    pipeline_context.register_qkvr(
                        data_repeat_idx=repeat_idx, loop_idx=0, data_offset=data_offset, fold_idx=fold,
                        qkv=qkv, r=r)
                else:
                    src_tgt = (rank_offset+1)%pp_size
                    pipeline_context.recv_qkvr_send_ro(
                        data_repeat_idx=repeat_idx, loop_idx=0, data_offset=data_offset-1, fold_idx=fold,
                        qkv_shape=pre_shapes, r_shape=r_shape, r_tensor=r, o_tensor=o, 
                        src_tgt=src_tgt, config=config)
                    
                    pipeline_context.register_o(
                        data_repeat_idx=repeat_idx, loop_idx=0, data_offset=data_offset, fold_idx=fold,
                        o=o)

                del qkv, r, o

        # param part
        for data_offset in range(1, pp_size):
            rank_offset = data_offset
            
            for fold in range(num_fold):
                if pp_rank == 0 and repeat_idx > 0:
                    parallel_state.set_virtual_pipeline_model_parallel_rank(num_loop)

                    r, o = pipeline_context.get_ro(
                        data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)

                    loss = forward_step(
                        forward_step_func, data_iterator, model, num_loop, (r, o), config)

                    pipeline_context.register_loss(repeat_idx, num_loop, data_offset, fold, loss)

                    del loss

                    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
                    pipeline_context.recv_ro(
                        data_repeat_idx=repeat_idx, loop_idx=0, data_offset=data_offset, fold_idx=fold,
                        r_shape=None, o_shape=None, src_tgt=None, config=config)
                    
                r, o = pipeline_context.get_ro(
                    data_repeat_idx=repeat_idx, loop_idx=0, data_offset=data_offset, fold_idx=fold)
                
                *qkv, r = forward_step(
                    forward_step_func, data_iterator, model, 0, (r, o), config)

                if data_offset == pp_size-1:
                    o = attn_func(qkv)

                    src_tgt = (pp_rank+pp_size-data_offset)%pp_size
                    if pp_rank == pp_size-1 and num_loop==1 and repeat_idx == num_data_repeat-1:
                        pipeline_context.send_ro(
                            data_repeat_idx=repeat_idx, loop_idx=0, data_offset=data_offset, fold_idx=fold,
                            r_tensor=r, o_tensor=o, src_tgt=src_tgt, config=config)
                    else:
                        pipeline_context.recv_qkvr_send_ro(
                            data_repeat_idx=repeat_idx, loop_idx=0, data_offset=data_offset-1, fold_idx=fold,
                            qkv_shape=pre_shapes, r_shape=r_shape, r_tensor=r, o_tensor=o, 
                            src_tgt=src_tgt, config=config)
                    
                    if pp_rank > 0 and args.checkpoint_without_attn:
                        pipeline_context.deallocate_output_tensor(r)

                    pipeline_context.register_ro(
                        data_repeat_idx=repeat_idx, loop_idx=0, data_offset=data_offset, fold_idx=fold,
                        r=r, o=o)
                else:
                    src_tgt = (pp_rank+data_offset+1)%pp_size
                    pipeline_context.recv_ro_send_qkvr(
                        data_repeat_idx=repeat_idx, loop_idx=0, data_offset=data_offset+1, fold_idx=fold,
                        r_shape=None if pp_rank == 0 and repeat_idx == 0 else r_shape,
                        o_shape=None if pp_rank == 0 and repeat_idx == 0 else o_shape,
                        qkv_tensor=qkv, r_tensor=r, src_tgt=src_tgt, config=config)
                    
                    pipeline_context.deallocate_output_tensor(qkv[0])
                    if pp_rank > 0 and args.checkpoint_without_attn:
                        pipeline_context.deallocate_output_tensor(r)

                    pipeline_context.register_qkvr(
                        data_repeat_idx=repeat_idx, loop_idx=0, data_offset=data_offset, fold_idx=fold,
                        qkv=qkv, r=r)
                
                del qkv, r, o

        # steady
        # print("="*25 + f" rank {pp_rank} repeat {repeat_idx} steady " + "="*25)
        for loop in range(1, num_loop):
            parallel_state.set_virtual_pipeline_model_parallel_rank(loop)

            # attn part
            for data_offset in reversed(range(pp_size-1)):
                rank_offset = pp_size - data_offset - 1

                for fold in range(num_fold):
                    qkv, r = pipeline_context.get_qkvr(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                    
                    o = attn_func(qkv)
                
                    if data_offset == 0:
                        *qkv, r = forward_step(
                            forward_step_func, data_iterator, model, loop, (r, o), config)

                        src_tgt = (pp_rank+data_offset+1)%pp_size
                        pipeline_context.recv_ro_send_qkvr(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset+1, fold_idx=fold,
                            r_shape=r_shape, o_shape=o_shape, qkv_tensor=qkv, r_tensor=r, 
                            src_tgt=src_tgt, config=config)
                        
                        pipeline_context.deallocate_output_tensor(qkv[0])
                        if args.checkpoint_without_attn:
                            pipeline_context.deallocate_output_tensor(r)

                        pipeline_context.register_qkvr(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold,
                            qkv=qkv, r=r)
                    else:
                        src_tgt = (pp_rank+rank_offset+1)%pp_size
                        pipeline_context.recv_qkvr_send_ro(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset-1, fold_idx=fold,
                            qkv_shape=pre_shapes, r_shape=r_shape, r_tensor=r, o_tensor=o, 
                            src_tgt=src_tgt, config=config)

                        pipeline_context.register_o(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold,
                            o=o)
                    
                    del qkv, r, o

            # param part
            for data_offset in range(1, pp_size):
                rank_offset = rank_offset

                for fold in range(num_fold):
                    r, o = pipeline_context.get_ro(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                    
                    *qkv, r = forward_step(
                        forward_step_func, data_iterator, model, loop, (r, o), config)

                    if data_offset == pp_size-1:
                        o = attn_func(qkv)

                        src_tgt = (pp_rank+pp_size-data_offset)%pp_size
                        if pp_rank == pp_size-1 and loop == num_loop-1 and repeat_idx == num_data_repeat-1:
                            pipeline_context.send_ro(
                                data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold,
                                r_tensor=r, o_tensor=o, src_tgt=src_tgt, config=config)
                        else: 
                            # this could possibly cross the data repeat, and affect repeat_idx in log, 
                            # but the communication should be ok
                            pipeline_context.recv_qkvr_send_ro(
                                data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset-1, fold_idx=fold,
                                qkv_shape=pre_shapes, r_shape=r_shape, r_tensor=r, o_tensor=o, 
                                src_tgt=src_tgt, config=config)
                        
                        if args.checkpoint_without_attn:
                            pipeline_context.deallocate_output_tensor(r)

                        pipeline_context.register_ro(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold,
                            r=r, o=o)
                    else:
                        src_tgt = (pp_rank+data_offset+1)%pp_size
                        pipeline_context.recv_ro_send_qkvr(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset+1, fold_idx=fold,
                            r_shape=r_shape, o_shape=o_shape, qkv_tensor=qkv, r_tensor=r,
                            src_tgt=src_tgt, config=config)
                        
                        pipeline_context.deallocate_output_tensor(qkv[0])
                        if args.checkpoint_without_attn:
                            pipeline_context.deallocate_output_tensor(r)

                        pipeline_context.register_qkvr(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold,
                            qkv=qkv, r=r)
                    
                    del qkv, r, o

        # cool down
        # print("="*25 + f" rank {pp_rank} repeat {repeat_idx} cool down " + "="*25)
        parallel_state.set_virtual_pipeline_model_parallel_rank(num_loop)
        # attn part
        for data_offset in reversed(range(pp_rank, pp_size-1)):
            rank_offset = pp_size - data_offset - 1

            for fold in range(num_fold):
                qkv, r = pipeline_context.get_qkvr(
                    data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)
                
                o = attn_func(qkv)

                if data_offset == 0:  # pp_rank == 0
                    loss = forward_step(
                        forward_step_func, data_iterator, model, num_loop, (r, o), config)

                    pipeline_context.register_loss(repeat_idx, num_loop, data_offset, fold, loss)

                    del loss
                    src_tgt = (pp_rank+data_offset+1)%pp_size  # rank_offset = data_offset + 1
                    if repeat_idx < num_data_repeat - 1:
                        parallel_state.set_virtual_pipeline_model_parallel_rank(0)
                        pipeline_context.recv_ro(
                            data_repeat_idx=repeat_idx+1, loop_idx=0, data_offset=0, fold_idx=fold,
                            r_shape=None, o_shape=None, src_tgt=None, config=config)
                        
                        r, o = pipeline_context.get_ro(
                            data_repeat_idx=repeat_idx+1, loop_idx=0, data_offset=0, fold_idx=fold)
                        
                        *qkv, r = forward_step(
                            forward_step_func, data_iterator, model, 0, (r, o), config)

                        pipeline_context.recv_ro_send_qkvr(
                            data_repeat_idx=repeat_idx, loop_idx=0, data_offset=data_offset+1, fold_idx=fold,
                            r_shape=r_shape, o_shape=o_shape, qkv_tensor=qkv, r_tensor=r, 
                            src_tgt=src_tgt, config=config)
                        
                        pipeline_context.deallocate_output_tensor(qkv[0])
                        
                        pipeline_context.register_qkvr(
                            data_repeat_idx=repeat_idx+1, loop_idx=0, data_offset=0, fold_idx=fold,
                            qkv=qkv, r=r)
                        parallel_state.set_virtual_pipeline_model_parallel_rank(num_loop)
                    else:
                        pipeline_context.recv_ro(
                            data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset+1, fold_idx=fold,
                            r_shape=r_shape, o_shape=o_shape, src_tgt=src_tgt, config=config)
                else:
                    src_tgt = (pp_rank+rank_offset+1)%pp_size
                    if data_offset == pp_rank and repeat_idx == num_data_repeat - 1:
                        pipeline_context.send_ro(
                            data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold,
                            r_tensor=r, o_tensor=o, src_tgt=src_tgt, config=config)
                    else:
                        pipeline_context.recv_qkvr_send_ro(
                            data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset-1, fold_idx=fold,
                            qkv_shape=pre_shapes, r_shape=r_shape, r_tensor=r, o_tensor=o, 
                            src_tgt=src_tgt, config=config)

                    pipeline_context.register_o(
                        data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold,
                        o=o)
                
                del qkv, r, o
                    
    for fold in range(num_fold):
        pipeline_context.clean_reqs(fold)

    if pp_rank == 0:
        parallel_state.set_virtual_pipeline_model_parallel_rank(num_loop)
        
        for data_offset in range(1, pp_size):

            for fold in range(num_fold):
                r, o = pipeline_context.get_ro(
                    data_repeat_idx=num_data_repeat-1, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)
                
                loss = forward_step(
                    forward_step_func, data_iterator, model, num_loop, (r, o), config)

                pipeline_context.register_loss(num_data_repeat-1, num_loop, data_offset, fold, loss)
            
                if data_offset < pp_size - 1:
                    src_tgt = (pp_rank+data_offset+1)%pp_size
                    pipeline_context.recv_ro(
                        data_repeat_idx=num_data_repeat-1, loop_idx=num_loop, data_offset=data_offset+1, fold_idx=fold,
                        r_shape=r_shape, o_shape=o_shape, src_tgt=src_tgt, config=config)
    
                del r, o, loss
    
    pipeline_context.check_reqs()
    pipeline_context.check_act_queue()
    if forward_only:
        assert not forward_only, "training only"
        return forward_data_store

    # backward
    # print("="*25 + f" rank {pp_rank} backward pass starts" + "="*25)
    assert parallel_state.get_virtual_pipeline_model_parallel_rank() == num_loop
    if pp_rank == 0:
        # print("="*25 + f" rank {pp_rank} reversed cool down loss backward " + "="*25)
        for data_offset in reversed(range(1, pp_size)):
            for fold in reversed(range(num_fold)):
                ro = pipeline_context.pop_input_ro(
                    data_repeat_idx=num_data_repeat-1, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)
                loss = pipeline_context.pop_output_loss(
                    data_repeat_idx=num_data_repeat-1, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)

                r_grad, o_grad = backward_step(
                    ro, loss, [None], config, model[0], args.fp16_lm_cross_entropy, 
                    num_microbatches, forward_data_store, collect_non_loss_data)

                src_tgt = (pp_rank+data_offset)%pp_size
                pipeline_context.send_ro_bwd(
                    data_repeat_idx=num_data_repeat-1, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold,
                    r_tensor=r_grad, o_tensor=o_grad, src_tgt=src_tgt, config=config)

                del ro, loss, r_grad, o_grad
    else:
        for fold in reversed(range(num_fold)):
            pipeline_context.recv_ro_bwd(
                data_repeat_idx=num_data_repeat-1, loop_idx=num_loop, data_offset=pp_rank, fold_idx=fold,
                r_shape=r_shape, o_shape=o_shape, src_tgt=0, config=config)

    for repeat_idx in reversed(range(num_data_repeat)):
        # cool down backward
        # print("="*25 + f" rank {pp_rank} repeat {repeat_idx} reversed cool down " + "="*25)
        parallel_state.set_virtual_pipeline_model_parallel_rank(num_loop)
        # attn part
        for data_offset in range(pp_rank, pp_size-1):
            rank_offset = pp_size - data_offset - 1
            src_tgt = (pp_rank + rank_offset)%pp_size

            for fold in reversed(range(num_fold)):
                if data_offset == 0: # pp rank = 0
                    if repeat_idx < num_data_repeat - 1:
                        parallel_state.set_virtual_pipeline_model_parallel_rank(0)
                        ro = pipeline_context.pop_input_ro(
                            data_repeat_idx=repeat_idx+1, loop_idx=0, data_offset=data_offset, fold_idx=fold)
                        qkvr = pipeline_context.pop_output_qkvr(
                            data_repeat_idx=repeat_idx+1, loop_idx=0, data_offset=data_offset, fold_idx=fold)
                        qkvr_grad = pipeline_context.pop_grad_qkvr(
                            data_repeat_idx=repeat_idx+1, loop_idx=0, data_offset=data_offset, fold_idx=fold)
                        
                        r_grad, o_grad = backward_step(ro, qkvr, qkvr_grad, config)
                        assert r_grad is None and o_grad is None

                        parallel_state.set_virtual_pipeline_model_parallel_rank(num_loop)

                        del ro, qkvr, qkvr_grad

                    qkvr = pipeline_context.pop_input_qkvr(
                        data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)
                    
                    loss = pipeline_context.pop_output_loss(
                        data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)
                    
                    *qkv_grad, r_grad = backward_step(
                        qkvr, loss, [None], config, model[0], args.fp16_lm_cross_entropy, 
                        num_microbatches, forward_data_store, collect_non_loss_data)

                    pipeline_context.recv_ro_send_qkvr_bwd(
                        data_repeat_idx=repeat_idx-1, loop_idx=num_loop, data_offset=data_offset+1, fold_idx=fold,
                        r_shape=r_shape, o_shape=o_shape, qkv_tensor=qkv_grad, r_tensor=r_grad, 
                        src_tgt=src_tgt, config=config)
                    
                    del qkvr, loss, qkv_grad, r_grad

                else:
                    qkv = pipeline_context.pop_input_qkv(
                        data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)
                    
                    o = pipeline_context.pop_output_o(
                        data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)
                    r_grad, o_grad = pipeline_context.pop_grad_ro(
                        data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)
                    
                    qkv_grad = backward_step(qkv, o, [o_grad], config)

                    pipeline_context.recv_ro_send_qkvr_bwd(
                        data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset+1, fold_idx=fold,
                        r_shape=r_shape, o_shape=o_shape, qkv_tensor=qkv_grad, r_tensor=r_grad,
                        src_tgt=src_tgt, config=config)
                    
                    del qkv, o, qkv_grad, r_grad, o_grad

        # steady backward
        # print("="*25 + f" rank {pp_rank} repeat {repeat_idx} reversed steady " + "="*25)
        for loop in reversed(range(1, num_loop)):
            # param part
            parallel_state.set_virtual_pipeline_model_parallel_rank(loop)
            for data_offset in reversed(range(1, pp_size)):
                rank_offset = data_offset
                for fold in reversed(range(num_fold)):
                    inp_ro = pipeline_context.pop_input_ro(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                    
                    if data_offset == pp_size - 1:
                        oup_ro = pipeline_context.pop_output_ro(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                        ro_grad = pipeline_context.pop_grad_ro(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)

                        r_grad, o_grad = backward_step(inp_ro, oup_ro, ro_grad, config)

                        del oup_ro, ro_grad
                    else:
                        qkvr = pipeline_context.pop_output_qkvr(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                        qkvr_grad = pipeline_context.pop_grad_qkvr(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)

                        r_grad, o_grad = backward_step(inp_ro, qkvr, qkvr_grad, config)

                        del qkvr, qkvr_grad
                    
                    src_tgt = (pp_rank+rank_offset)%pp_size
                    pipeline_context.recv_qkvr_send_ro_bwd(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset-1, fold_idx=fold,
                        qkv_shape=pre_shapes, r_shape=r_shape, r_tensor=r_grad, o_tensor=o_grad,
                        src_tgt=src_tgt, config=config)

                    del inp_ro, r_grad, o_grad
                    
            # attn part
            for data_offset in range(pp_size-1):
                rank_offset = pp_size - data_offset - 1
                src_tgt = (pp_rank+rank_offset)%pp_size

                for fold in reversed(range(num_fold)):
                    if data_offset == 0:
                        inp_qkvr = pipeline_context.pop_input_qkvr(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                        oup_qkvr = pipeline_context.pop_output_qkvr(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                        qkvr_grad = pipeline_context.pop_grad_qkvr(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)

                        *qkv_grad, r_grad = backward_step(inp_qkvr, oup_qkvr, qkvr_grad, config)

                        del inp_qkvr, oup_qkvr, qkvr_grad
                    else:
                        qkv = pipeline_context.pop_input_qkv(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                        o = pipeline_context.pop_output_o(
                            data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)
                        r_grad, o_grad = pipeline_context.pop_grad_ro(
                            data_repeat_idx=repeat_idx, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)
                        
                        qkv_grad = backward_step(qkv, o, [o_grad], config)
                        
                        del qkv, o, o_grad

                    pipeline_context.recv_ro_send_qkvr_bwd(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset+1, fold_idx=fold,
                        r_shape=r_shape, o_shape=o_shape, qkv_tensor=qkv_grad, r_tensor=r_grad,
                        src_tgt=src_tgt, config=config)

                    del qkv_grad, r_grad

        # warm up backward
        loop = 0
        # print("="*25 + f" rank {pp_rank} repeat {repeat_idx} reversed warm up " + "="*25)
        parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        # param part
        for data_offset in reversed(range(1, pp_size)):
            rank_offset = data_offset
            src_tgt = (pp_rank+rank_offset)%pp_size

            for fold in reversed(range(num_fold)):
                inp_ro = pipeline_context.pop_input_ro(
                    data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                
                if data_offset == pp_size - 1:
                    oup_ro = pipeline_context.pop_output_ro(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                    ro_grad = pipeline_context.pop_grad_ro(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)

                    r_grad, o_grad = backward_step(inp_ro, oup_ro, ro_grad, config)

                    del inp_ro, oup_ro, ro_grad
                else:
                    qkvr = pipeline_context.pop_output_qkvr(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                    qkvr_grad = pipeline_context.pop_grad_qkvr(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)

                    r_grad, o_grad = backward_step(inp_ro, qkvr, qkvr_grad, config)
                    del inp_ro, qkvr, qkvr_grad
                if pp_rank == 0:
                    assert r_grad is None and o_grad is None

                    if repeat_idx > 0:
                        parallel_state.set_virtual_pipeline_model_parallel_rank(num_loop)
                        ro = pipeline_context.pop_input_ro(
                            data_repeat_idx=repeat_idx-1, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)
                        loss = pipeline_context.pop_output_loss(
                            data_repeat_idx=repeat_idx-1, loop_idx=num_loop, data_offset=data_offset, fold_idx=fold)

                        r_grad, o_grad = backward_step(
                            ro, loss, [None], config, model[0], args.fp16_lm_cross_entropy, 
                            num_microbatches, forward_data_store, collect_non_loss_data)

                        pipeline_context.recv_qkvr_send_ro_bwd(
                            data_repeat_idx=repeat_idx-1, loop_idx=loop, data_offset=data_offset-1, fold_idx=fold,
                            qkv_shape=pre_shapes, r_shape=r_shape, r_tensor=r_grad, o_tensor=o_grad,
                            src_tgt=src_tgt, config=config)
                        parallel_state.set_virtual_pipeline_model_parallel_rank(0)
                        
                        del ro, loss
                    else:
                        pipeline_context.recv_qkvr_bwd(
                            data_repeat_idx=repeat_idx-1, loop_idx=loop, data_offset=data_offset-1, fold_idx=fold,
                            qkv_shape=pre_shapes, r_shape=r_shape,src_tgt=src_tgt, config=config)
                else:
                    pipeline_context.recv_qkvr_send_ro_bwd(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset-1, fold_idx=fold,
                        qkv_shape=pre_shapes, r_shape=r_shape, r_tensor=r_grad, o_tensor=o_grad,
                        src_tgt=src_tgt, config=config)
                del r_grad, o_grad
        
        # attn part
        for data_offset in range(pp_rank):
            rank_offset = pp_size - data_offset - 1
            src_tgt = (pp_rank+rank_offset)%pp_size

            for fold in reversed(range(num_fold)):
                if data_offset == 0:
                    inp_qkvr = pipeline_context.pop_input_qkvr(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                    oup_qkvr = pipeline_context.pop_output_qkvr(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                    qkvr_grad = pipeline_context.pop_grad_qkvr(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)

                    *qkv_grad, r_grad = backward_step(inp_qkvr, oup_qkvr, qkvr_grad, config)

                    if pp_rank > 1 or repeat_idx > 0:
                        pipeline_context.recv_ro_send_qkvr_bwd(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset+1, fold_idx=fold,
                            r_shape=r_shape, o_shape=o_shape, qkv_tensor=qkv_grad, r_tensor=r_grad,
                            src_tgt=src_tgt, config=config)
                    else:
                        pipeline_context.send_qkvr_bwd(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold,
                            qkv_tensor=qkv_grad, r_tensor=r_grad, src_tgt=src_tgt, config=config)
                    
                    del inp_qkvr, oup_qkvr, qkvr_grad, qkv_grad, r_grad
                else:
                    qkv = pipeline_context.pop_input_qkv(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                    o = pipeline_context.pop_output_o(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
                    r_grad, o_grad = pipeline_context.pop_grad_ro(
                        data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)

                    qkv_grad = backward_step(qkv, o, [o_grad], config)

                    if data_offset < pp_rank - 1 or repeat_idx > 0:
                        pipeline_context.recv_ro_send_qkvr_bwd(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset+1, fold_idx=fold,
                            r_shape=r_shape, o_shape=o_shape, qkv_tensor=qkv_grad, r_tensor=r_grad,
                            src_tgt=src_tgt, config=config)
                    else:
                        pipeline_context.send_qkvr_bwd(
                            data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold,
                            qkv_tensor=qkv_grad, r_tensor=r_grad, src_tgt=src_tgt, config=config)

                    del qkv, o, r_grad, o_grad, qkv_grad
    
    if pp_rank == 0:
        parallel_state.set_virtual_pipeline_model_parallel_rank(num_loop)
        # finish the fist part of param at warm up phase
        data_offset = 0
        repeat_idx = 0
        for fold in range(num_fold):
            ro = pipeline_context.pop_input_ro(
                data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
            qkvr = pipeline_context.pop_output_qkvr(
                data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)
            qkvr_grad = pipeline_context.pop_grad_qkvr(
                data_repeat_idx=repeat_idx, loop_idx=loop, data_offset=data_offset, fold_idx=fold)

            r_grad, o_grad = backward_step(ro, qkvr, qkvr_grad, config)
            assert r_grad is None and o_grad is None

            del ro, qkvr, qkvr_grad

    for fold in range(num_fold):
        pipeline_context.clean_reqs(fold)
    pipeline_context.check_context()
    # torch.distributed.barrier()
    # exit(0)
    return forward_data_store
