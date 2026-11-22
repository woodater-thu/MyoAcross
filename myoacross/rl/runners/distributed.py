"""
Synchronous data parallel helpers; ordinary Python runs remain single-process.

Launch RL with torchrun. Each rank owns its simulation and rollout storage;
only gradients and small statistics cross ranks, never simulation state.
"""

from contextlib import contextmanager
from datetime import timedelta
import os

import torch
import torch.distributed as dist


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def is_primary() -> bool:
    return rank() == 0


@contextmanager
def training_process(device: str):
    """Bind torchrun's local device before any CUDA/Warp initialization."""
    size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    selected = torch.device(device)
    if selected.type == "cuda":
        if "LOCAL_RANK" in os.environ:
            selected = torch.device("cuda", local_rank)
        torch.cuda.set_device(selected)
    elif selected.type != "cpu" and size > 1:
        raise ValueError("Distributed training requires CUDA or CPU")

    owns_group = size > 1 and not dist.is_initialized()
    if owns_group:
        # Allow time for shared preparation before all ranks enter training.
        dist.init_process_group(
            backend="nccl" if selected.type == "cuda" else "gloo",
            timeout=timedelta(hours=1),
        )
    try:
        yield str(selected)
    finally:
        if owns_group:
            dist.destroy_process_group()


def primary_call(function):
    """Run shared-filesystem work once and propagate its result or failure."""
    if world_size() == 1:
        return function()
    message = [None]
    if is_primary():
        try:
            message[0] = (True, function())
        except Exception as error:
            message[0] = (False, f"{type(error).__name__}: {error}")
    dist.broadcast_object_list(message, src=0)
    success, value = message[0]
    if not success:
        raise RuntimeError(f"Rank zero failed: {value}")
    return value


@torch.no_grad()
def broadcast_model(model: torch.nn.Module) -> None:
    if world_size() > 1:
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=0)


@torch.no_grad()
def mean(tensor: torch.Tensor) -> torch.Tensor:
    result = tensor.clone()
    if world_size() > 1:
        dist.all_reduce(result)
        result.div_(world_size())
    return result


@torch.no_grad()
def require_finite(tensor: torch.Tensor, name: str) -> None:
    valid = torch.isfinite(tensor).all().to(dtype=torch.int32)
    if world_size() > 1:
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        raise FloatingPointError(f"Non-finite {name} on at least one rank")


@torch.no_grad()
def moments(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Population moments over all ranks, including between-rank variance.

    Gather only count/mean/variance, then use a parallel variance merge to avoid
    subtracting large squared means. All ranks participate, even for empty inputs.
    """
    count = samples.new_tensor(float(samples.shape[0]))
    if samples.shape[0]:
        variance, average = torch.var_mean(samples, dim=0, correction=0)
    else:
        average = samples.new_zeros(samples.shape[1:])
        variance = torch.zeros_like(average)
    if world_size() == 1:
        return count, average, variance

    record = torch.cat((count.reshape(1), average.flatten(), variance.flatten()))
    records = [torch.empty_like(record) for _ in range(world_size())]
    dist.all_gather(records, record)
    records = torch.stack(records)
    width = average.numel()
    counts = records[:, :1]
    averages = records[:, 1:1 + width]
    variances = records[:, 1 + width:]
    total = counts.sum()
    combined_mean = (counts * averages).sum(0) / total.clamp_min(1)
    combined_var = (
        counts * (variances + (averages - combined_mean).square())
    ).sum(0) / total.clamp_min(1)
    return total, combined_mean.reshape_as(average), combined_var.reshape_as(variance)


@torch.no_grad()
def average_gradients(parameters: list[torch.nn.Parameter]) -> None:
    """Average before clipping/Adam; preserve globally unused parameters.

The model uses a shared encoder and optional FiLM branches. A missing
gradient must stay None on all ranks so Adam does not update unused weights.
"""
    if world_size() == 1:
        return
    active = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        dtype=torch.int32, device=parameters[0].device,
    )
    dist.all_reduce(active)
    if ((active != 0) & (active != world_size())).any().item():
        raise RuntimeError("Trainable parameter usage differs across ranks")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    dist.all_reduce(flat)
    flat.div_(world_size())
    offset = 0
    for gradient in gradients:
        gradient.copy_(flat[offset:offset + gradient.numel()].view_as(gradient))
        offset += gradient.numel()
