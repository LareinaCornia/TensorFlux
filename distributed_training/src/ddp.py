import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torchvision.models import vgg16
import json
import os

from . import utils

class DistributedDataParallel():
    """
    A module implementation of Distributed Data Parallel (DDP)
    
    Attributes:
        - model (nn.Module): the underlying model to be trained
        - device (torch.device): the device the model is located on for this rank
        - rank (int): the rank of the process the module is located on
        - world_size (int): the total number of processes running DDP
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.broadcast_parameters()

    @torch.no_grad()
    def broadcast_parameters(self):
        for parameter in self.model.parameters():
            dist.broadcast(parameter, src=0)

    @torch.no_grad()
    def average_gradients(self):
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue

            dist.all_reduce(
                parameter.grad,
                op=dist.ReduceOp.SUM,
            )
            parameter.grad.div_(self.world_size)


def train_vgg16_cifar10_ddp_worker(
    rank: int, world_size: int, train_dataset: Dataset, stats_queue: mp.Queue, 
    num_warmup_batches: int = 0,
    num_batches: int = 5, cores_per_rank: int = 1, 
    batch_size: int = 32, learning_rate: float = 1e-2,
    check_weights: bool = False, check_output: bool = True
):
    """
    For a given worker process, trains a VGG16 model on the CIFAR-10 
    dataset using DDP.
    
    Args:
        rank (int): the rank of the current process
        world_size (int): the total number of processes
        train_dataset (Dataset): the training dataset
        stats_queue (mp.Queue): the queue for communicating statistics about this worker
        num_batches (int): the number of batches to train for; defaults to 5
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        batch_size (int): the number of data points processed in one step of
        training; defaults to 32
        learning_rate (float): the learning rate of the optimizer; defaults to 1e-2
        check_weights (bool): boolean to determine whether weights are saved;
        defaults to False
        check_output (bool): boolean to determine whether the model's output is
        saved; defaults to True
    """
    stats = {
        utils.RANK: rank,
        utils.COMP_TIME: 0.0,
        utils.COMM_TIME: 0.0,
        utils.OPT_TIME: 0.0,
        utils.TOTAL_TIME: 0.0,
    }
    
    utils.debug_print(f"Initializing DDP on rank {rank}.")
    device = utils.parallel_setup(rank, world_size)
    utils.pin_to_core(rank, cores_per_rank)
    utils.seed_everything(1390)

    # set up distributed data loader
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=distributed_sampler,
        shuffle=False,
        num_workers=world_size
    )

    # load vgg16 model
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 10)
    layers = [layer for layer in model.features] + [model.avgpool, nn.Flatten()] + [layer for layer in model.classifier]
    model = nn.Sequential(*layers)
    model = model.to(device)
    model = DistributedDataParallel(model, device)
    model.model.eval()

    # define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.model.parameters(), lr=learning_rate)

    memory_log = []
    def record_memory(batch_idx, phase):
        if device.type == "cuda" and batch_idx < num_warmup_batches:
            entry = utils.log_memory(
                device, rank, batch_idx, -1, phase
            )
            if entry:
                memory_log.append(entry)

    total_num_batches = num_batches + num_warmup_batches

    for i, (inputs, labels) in enumerate(train_loader):
        if i == total_num_batches:
            inputs = inputs.to(device)
            break
        
        # start
        utils.sync_if_cuda(device)
        step_start = time.perf_counter()

        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        record_memory(i, "after_zero_grad")

        # computation
        utils.sync_if_cuda(device)
        computation_start = time.perf_counter()

        outputs = model.model(inputs)
        loss = loss_fn(outputs, labels)

        record_memory(i, "after_forward")
        loss.backward()
        record_memory(i, "after_backward")

        utils.sync_if_cuda(device)
        computation_end = time.perf_counter()

        # communication
        model.average_gradients()
        record_memory(i, "after_gradient_sync")

        utils.sync_if_cuda(device)
        communication_end = time.perf_counter()

        # update
        optimizer.step()
        record_memory(i, "after_optimizer_step")

        utils.sync_if_cuda(device)
        optimizer_end = time.perf_counter()

        if i >= num_warmup_batches:
            stats[utils.COMP_TIME] += (
                computation_end - computation_start
            )
            stats[utils.COMM_TIME] += (
                communication_end - computation_end
            )
            stats[utils.OPT_TIME] += (
                optimizer_end - communication_end
            )
            stats[utils.TOTAL_TIME] += (
                optimizer_end - step_start
            )

    for key in ( utils.COMP_TIME, utils.COMM_TIME, utils.OPT_TIME ):
        stats[key] /= num_batches

    if check_weights:
        torch.save(model.model.state_dict(), utils.STATE_DICT_DIR / f'rank_{rank}_weights.pt')
    if check_output:
        print(f"Saving model's output on rank {rank}")
        torch.save(model.model(inputs).cpu(), utils.SAVED_TENSOR_DIR / f'rank_{rank}_test_output.pt')
    if len(memory_log) > 0:
        utils.MEMORY_LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = utils.MEMORY_LOG_DIR / f'ddp{world_size}_rank{rank}_memory_log.pt'
        with path.open('w') as f:
            json.dump(memory_log, f)

    # Evaluate the model on the test dataset
    if rank == 0:
        print("Finished training")

    stats_queue.put(stats)
    utils.parallel_cleanup()

def train_vgg16_cifar10_ddp(
    world_size: int = 1, num_batches: int = 5, batch_size: int = 32, 
    learning_rate: float = 1e-2, cores_per_rank: int = 1, 
    check_weights: bool = False, check_output: bool = True
) -> dict:
    """
    Trains a VGG16 model on the CIFAR-10 dataset using DDP.
    
    Args:
        world_size (int): the number of processes to train with; defaults to 1
        num_batches (int): the number of batches to train for; defaults to 5
        batch_size (int): the number of data points processed in one step of
        training; defaults to 32
        learning_rate (float): the optimizer's learning rate; defaults to 1e-2
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        check_weights (bool): boolean to determine whether weights are saved;
        defaults to False
        check_output (bool): boolean to determine whether the model's output is
        saved; defaults to True
    """
    stats_queue = mp.Queue()
    num_warmup_batches = 0
    if torch.cuda.is_available():
        num_warmup_batches = 3
    mp.spawn(
        train_vgg16_cifar10_ddp_worker, 
        args=(world_size, utils.get_train_dataset(), stats_queue, 
              num_warmup_batches,
              num_batches, 
              cores_per_rank, batch_size, learning_rate, check_weights, check_output), 
        nprocs=world_size, join=True
    )
    return utils.agg_stats_per_rank(stats_queue)
