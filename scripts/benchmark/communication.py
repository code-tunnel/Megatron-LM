import os

import torch
import torch.distributed as dist

from benchmark_utils import dist_benchmark


ROW_GROUP = None
COL_GROUP = None


def p2p(tensor, group):
    rank = dist.get_rank(group)
    
    ops = []
    if rank % 2 == 0:
        ops.append(dist.P2POp(dist.isend, tensor, dist.get_global_rank(group, rank+1), group))
    else:
        ops.append(dist.P2POp(dist.irecv, tensor, dist.get_global_rank(group, rank-1), group))

    reqs = dist.batch_isend_irecv(ops)
    for each in reqs:
        each.wait()


def allreduce(tensor, group):
    dist.all_reduce(tensor, group=group)


def init_local_group(nodes, gpus_per_node, rank):
    """organize the pg as rows and cols"""
    global ROW_GROUP, COL_GROUP

    if gpus_per_node > 1:
        for n in range(nodes):
            ranks = list(range(n*gpus_per_node, (n+1)*gpus_per_node))
            row_group = dist.new_group(ranks)
            if rank in ranks:
                ROW_GROUP = row_group

    if nodes > 1:
        for g in range(gpus_per_node):
            ranks = [g + i * gpus_per_node for i in range(nodes)]
            col_group = dist.new_group(ranks)
            if rank in ranks:
                COL_GROUP = col_group


def main():
    # assert "SLURM_JOB_NUM_NODES" in os.environ
    # assert "LOCAL_WORLD_SIZE" in os.environ
    # assert "RANK" in os.environ
    # assert "WORLD_SIZE" in os.environ

    nodes = int(os.getenv("SLURM_JOB_NUM_NODES", os.getenv("NNODES")))
    gpus_per_node = int(os.getenv("LOCAL_WORLD_SIZE", os.getenv("GPUS_PER_NODE")))

    rank = int(os.getenv('RANK', os.getenv('OMPI_COMM_WORLD_RANK')))
    world_size = int(os.getenv("WORLD_SIZE", os.getenv("OMPI_COMM_WORLD_SIZE")))

    local_rank = rank % torch.cuda.device_count()

    dist.init_process_group(
        rank=rank, world_size=world_size, backend='nccl')
    
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    print(f"global init {rank}/{world_size}, local: {local_rank}, "
          f"nnodes: {nodes}, gpus_per_node: {gpus_per_node}, "
          f"{os.getenv('CUDA_VISIBLE_DEVICES')}, {torch.cuda.device_count()}")

    # init local row and col groups
    init_local_group(nodes, gpus_per_node, rank)

    # preprare data
    data_size = 1024*1024*1024  #  data_size* 4 MB
    data_volume = data_size * 4 / 1024**3
    tensor = torch.rand(data_size, device=device, dtype=torch.float)
    
    # test row: gpus within a node
    if gpus_per_node > 1:
        assert ROW_GROUP is not None
        row_rank = dist.get_rank(ROW_GROUP)

        std, avg = dist_benchmark(p2p, (tensor, ROW_GROUP), warm_up=10, iters=10, group=ROW_GROUP)

        print(f">>> rank {row_rank} in row p2p within node: {(data_volume / (avg/1000)):.2f} GiB/s, std: {std:.2f}")
        
        std, avg = dist_benchmark(allreduce, (tensor, ROW_GROUP), warm_up=10, iters=10, group=ROW_GROUP)

        print(f">>> rank {row_rank} in row allreduce within node: {(data_volume / (avg/1000)):.2f} GiB/s, std: {std:.2f}")
        
    if nodes > 1:
        assert COL_GROUP is not None
        col_rank = dist.get_rank(COL_GROUP)

        std, avg = dist_benchmark(p2p, (tensor, COL_GROUP), warm_up=10, iters=10, group=COL_GROUP)

        print(f">>> rank {col_rank} in col p2p across nodes: {(data_volume / (avg/1000)):.2f} GiB/s, std: {std:.2f}",)

        std, avg = dist_benchmark(allreduce, (tensor, COL_GROUP), warm_up=10, iters=10, group=COL_GROUP)

        print(f">>> rank {row_rank} in col allreduce across node: {(data_volume / (avg/1000)):.2f} GiB/s, std: {std:.2f}")
        

if __name__ == "__main__":
    main()
