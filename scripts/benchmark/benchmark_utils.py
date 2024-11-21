import time

import torch
import torch.distributed as dist


def benchmark(func, func_inputs, warm_up, iters, 
              excluded_func=None, exfunc_inputs=None):
    
    for _ in range(warm_up):
        if excluded_func is not None:
            excluded_func(*exfunc_inputs)

        func(*func_inputs)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    # elapsed = []
    for i in range(iters):
        if excluded_func is not None:
            excluded_func(*exfunc_inputs)

        # torch.cuda.synchronize()
        # start = time.time()
        start_events[i].record()
        func(*func_inputs)
        end_events[i].record()
        # torch.cuda.synchronize()
        # elapsed.append((time.time() - start)*1000)
    
    torch.cuda.synchronize()
    elapsed = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    elapsed = torch.tensor(elapsed, dtype=torch.double)
    std, avg = torch.std_mean(elapsed)
    return std.item(), avg.item()


def dist_benchmark(func, func_inputs, warm_up, iters, group=None,
                   excluded_func=None, exfunc_inputs=None):
    """test `func` with distributed setting, unit is in ms"""
    
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.cuda.current_device() if dist.get_backend(group) == 'nccl' else None

    for _ in range(warm_up):
        if excluded_func is not None:
            excluded_func(*exfunc_inputs)

        func(*func_inputs)
        torch.cuda.synchronize()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    # elapsed = []
    for i in range(iters):
        if excluded_func is not None:
            excluded_func(*exfunc_inputs)
        # dist.barrier(group)

        # torch.cuda.synchronize()
        # start = time.time()
        start_events[i].record()
        func(*func_inputs)
        end_events[i].record()
        torch.cuda.synchronize()
        # torch.cuda.synchronize()
        # elapsed.append((time.time() - start)*1000)

    torch.cuda.synchronize()
    elapsed = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    elapsed = torch.tensor(elapsed, dtype=torch.double, device=device)
    gathered = [torch.empty_like(elapsed) if i !=rank else elapsed \
                for i in range(world_size)]
    dist.all_gather(gathered, elapsed, group)

    gathered = torch.stack(gathered, dim=0)  # world size, iters
    max_val = gathered.max(dim=0, keepdims=False).values # iters
    std, avg = torch.std_mean(max_val)
    return std.item(), avg.item()


def fwd_bwd_benchmark(fwd_func, fwd_inputs, bwd_func, bwd_inputs, warm_up, iters, excluded_func=None, exfunc_inputs=None):
    for _ in range(warm_up):
        if excluded_func is not None:
            excluded_func(*exfunc_inputs)
        
        outputs = fwd_func(*fwd_inputs)
        bwd_func(outputs, *bwd_inputs)
    
    fwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    fwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    bwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    bwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    # fwd_elapsed, bwd_elapsed = [], []
    for i in range(iters):
        if excluded_func is not None:
            excluded_func(*exfunc_inputs)

        # torch.cuda.synchronize()
        # start = time.time()
        fwd_start_events[i].record()
        outputs = fwd_func(*fwd_inputs)
        fwd_end_events[i].record()
        # torch.cuda.synchronize()
        # fwd_elapsed.append((time.time() - start)*1000)

        # torch.cuda.synchronize()
        # start = time.time()
        bwd_start_events[i].record()
        bwd_func(outputs, *bwd_inputs)
        bwd_end_events[i].record()
        # torch.cuda.synchronize()
        # bwd_elapsed.append((time.time() - start)*1000)
    
    # 2, iters
    torch.cuda.synchronize()
    fwd_elapsed = [s.elapsed_time(e) for s, e in zip(fwd_start_events, fwd_end_events)]
    bwd_elapsed = [s.elapsed_time(e) for s, e in zip(bwd_start_events, bwd_end_events)]
    elapsed = torch.tensor(
        [fwd_elapsed, bwd_elapsed], dtype=torch.double)
    # 2
    std, avg = torch.std_mean(elapsed, dim=1)
    std, avg = std.tolist(), avg.tolist()
    return (std[0], avg[0]), (std[1], avg[1])


def dist_fwd_bwd_benchmark(fwd_func, fwd_inputs, bwd_func, bwd_inputs, 
                           warm_up, iters, excluded_func=None, exfunc_inputs=None, group=None):
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.cuda.current_device() if dist.get_backend(group) == 'nccl' else None

    for _ in range(warm_up):
        if excluded_func is not None:
            excluded_func(*exfunc_inputs)
        
        outputs = fwd_func(*fwd_inputs)
        bwd_func(outputs, *bwd_inputs)

    fwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    fwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    bwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    bwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    # fwd_elapsed, bwd_elapsed = [], []
    for i in range(iters):
        if excluded_func is not None:
            excluded_func(*exfunc_inputs)
        # barrier here will cause non-master nodes stuck
        # dist.barrier(group)

        # torch.cuda.synchronize()
        # start = time.time()
        fwd_start_events[i].record()
        outputs = fwd_func(*fwd_inputs)
        fwd_end_events[i].record()
        # torch.cuda.synchronize()
        # fwd_elapsed.append((time.time() - start)*1000)

        # torch.cuda.synchronize()
        # start = time.time()
        bwd_start_events[i].record()
        bwd_func(outputs, *bwd_inputs)
        bwd_end_events[i].record()
        # torch.cuda.synchronize()
        # bwd_elapsed.append((time.time() - start)*1000)

    # 2, iters
    torch.cuda.synchronize()
    fwd_elapsed = [s.elapsed_time(e) for s, e in zip(fwd_start_events, fwd_end_events)]
    bwd_elapsed = [s.elapsed_time(e) for s, e in zip(bwd_start_events, bwd_end_events)]
    elapsed = torch.tensor(
        [fwd_elapsed, bwd_elapsed], dtype=torch.double, device=device)

    gathered = [torch.empty_like(elapsed) for _ in range(world_size)]
    dist.all_gather(gathered, elapsed, group)
    gathered = torch.stack(gathered, dim=0)  # world size, 2, iters
    
    max_val = gathered.max(dim=0, keepdims=False).values # 2, iters
    std, avg = torch.std_mean(max_val, dim=1) # 2
    std, avg = std.tolist(), avg.tolist()
    return (std[0], avg[0]), (std[1], avg[1])


def profile(func, func_inputs, warm_up, iters, trace_path,
              excluded_func=None, exfunc_inputs=None):
    prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                skip_first=0, wait=0, warmup=warm_up, active=iters, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
            profile_memory=True,
        )

    prof.start()

    for _ in range(warm_up+iters):
        if excluded_func is not None:
            excluded_func(*exfunc_inputs)
        
        func(*func_inputs)

        prof.step()
    
    prof.stop()
