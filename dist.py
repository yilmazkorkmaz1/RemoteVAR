"""Small distributed-runtime helper used by the VAR model modules."""

import datetime
import functools
import os
import sys
from typing import List, Union

import torch
import torch.distributed as tdist
import torch.multiprocessing as mp


__rank = 0
__local_rank = 0
__world_size = 1
__device = "cuda" if torch.cuda.is_available() else "cpu"
__initialized = False


def initialized():
    return __initialized


def initialize(fork=False, backend="nccl", gpu_id_if_not_distibuted=0, timeout=30):
    global __device, __rank, __local_rank, __world_size, __initialized

    if not torch.cuda.is_available():
        print("[dist initialize] cuda is not available, use cpu instead", file=sys.stderr)
        return
    if "RANK" not in os.environ:
        torch.cuda.set_device(gpu_id_if_not_distibuted)
        __device = torch.empty(1).cuda().device
        print(f'[dist initialize] env variable "RANK" is not set, use {__device} as the device', file=sys.stderr)
        return

    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", global_rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)

    if mp.get_start_method(allow_none=True) is None:
        method = "fork" if fork else "spawn"
        mp.set_start_method(method)

    tdist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=timeout * 60))
    __local_rank = local_rank
    __rank = tdist.get_rank()
    __world_size = tdist.get_world_size()
    __device = torch.empty(1).cuda().device
    __initialized = True


def get_rank():
    return __rank


def get_local_rank():
    return __local_rank


def get_world_size():
    return __world_size


def get_device():
    return __device


def set_gpu_id(gpu_id: int):
    if gpu_id is None:
        return
    if not isinstance(gpu_id, (str, int)):
        raise NotImplementedError
    global __device
    torch.cuda.set_device(int(gpu_id))
    __device = torch.empty(1).cuda().device


def is_master():
    return __rank == 0


def is_local_master():
    return __local_rank == 0


def is_visualizer():
    return __rank == 0


def new_group(ranks: List[int]):
    return tdist.new_group(ranks=ranks) if __initialized else None


def barrier():
    if __initialized:
        tdist.barrier()


def allreduce(tensor: torch.Tensor, async_op=False):
    if not __initialized:
        return None
    if tensor.is_cuda:
        return tdist.all_reduce(tensor, async_op=async_op)
    cuda_tensor = tensor.detach().cuda()
    result = tdist.all_reduce(cuda_tensor, async_op=async_op)
    tensor.copy_(cuda_tensor.cpu())
    return result


def allgather(tensor: torch.Tensor, cat=True) -> Union[List[torch.Tensor], torch.Tensor]:
    if __initialized:
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        tensors = [torch.empty_like(tensor) for _ in range(__world_size)]
        tdist.all_gather(tensors, tensor)
    else:
        tensors = [tensor]
    return torch.cat(tensors, dim=0) if cat else tensors


def allgather_diff_shape(tensor: torch.Tensor, cat=True) -> Union[List[torch.Tensor], torch.Tensor]:
    if not __initialized:
        tensors = [tensor]
        return torch.cat(tensors, dim=0) if cat else tensors

    if not tensor.is_cuda:
        tensor = tensor.cuda()
    tensor_size = torch.tensor(tensor.size(), device=tensor.device)
    gathered_sizes = [torch.empty_like(tensor_size) for _ in range(__world_size)]
    tdist.all_gather(gathered_sizes, tensor_size)

    max_batch = max(size[0].item() for size in gathered_sizes)
    padding = max_batch - tensor_size[0].item()
    if padding:
        tensor = torch.cat((tensor, tensor.new_empty((padding, *tensor.size()[1:]))), dim=0)

    padded = [torch.empty_like(tensor) for _ in range(__world_size)]
    tdist.all_gather(padded, tensor)
    tensors = [value[: size[0].item()] for value, size in zip(padded, gathered_sizes)]
    return torch.cat(tensors, dim=0) if cat else tensors


def broadcast(tensor: torch.Tensor, src_rank) -> None:
    if not __initialized:
        return
    if tensor.is_cuda:
        tdist.broadcast(tensor, src=src_rank)
        return
    cuda_tensor = tensor.detach().cuda()
    tdist.broadcast(cuda_tensor, src=src_rank)
    tensor.copy_(cuda_tensor.cpu())


def dist_fmt_vals(val: float, fmt: Union[str, None] = "%.2f") -> Union[torch.Tensor, List]:
    if not initialized():
        return torch.tensor([val]) if fmt is None else [fmt % val]
    values = torch.zeros(__world_size)
    values[__rank] = val
    allreduce(values)
    if fmt is None:
        return values
    return [fmt % value for value in values.cpu().numpy().tolist()]


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs) if kwargs.pop("force", False) or is_master() else None
        barrier()
        return result

    return wrapper


def local_master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs) if kwargs.pop("force", False) or is_local_master() else None
        barrier()
        return result

    return wrapper


def for_visualize(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs) if is_visualizer() else None

    return wrapper


def finalize():
    global __initialized
    if __initialized:
        tdist.destroy_process_group()
        __initialized = False
