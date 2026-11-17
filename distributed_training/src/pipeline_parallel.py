import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import torch.distributed as dist
import torch.multiprocessing as mp

from . import utils

class PipelineParallel():
    """
    A module implementation of pipeline parallelism.
    
    Attributes:
        - device (torch.device): the device for this model to reside on
        - rank (int): the rank of the process the module is located on
        - world_size (int): the total number of processes running model parallelism
        - worker (ModelParallelWorker): the underlying class responsible for this
          rank's partitioned part of the model
        - loss_fn (nn.Module): the loss function to use when training
        - part_comm_info (dict[str, torch.Size]): the results of communication 
          analysis, storing the activation size and gradient size for this partition
        - num_microbatches (int): the number of microbatches to split each batch into
    """
    def __init__(self, partition: list[nn.Module],
                 device: torch.device, part_comm_info: dict[str, torch.Size],
                 batch_times: list[float], learning_rate: float = 0.001, 
                 loss_fn: nn.Module = nn.CrossEntropyLoss(), num_microbatches: int = 2):
        self.partition = partition
        self.batch_times = batch_times
        self.learning_rate = learning_rate
        self.device = device
        for layer in self.partition:
            layer.to(self.device)
            layer.eval()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.loss_fn = loss_fn
        self.part_comm_info = part_comm_info
        self.num_microbatches = num_microbatches
        self.optimizer = torch.optim.SGD(
            params=(p for layer in self.partition for p in layer.parameters()),
            lr=self.learning_rate
        )
        self.activations: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {
            idx: {} for idx in range(self.num_microbatches)
        }

    def forward(self, input_tensor: torch.Tensor | None, 
                microbatch_idx: int, is_warmup: bool = False) -> torch.Tensor:
        """
        Performs the forward pass of the model for this rank.

        Args:
            input_tensor (torch.Tensor | None): the input tensor to the full model,
            if this rank is responsible for the first layers of the model;
            otherwise None
            microbatch_idx (int): the ID of current microbatch within a given batch
            is_warmup (bool): Whether this is a warmup batch or not
        Returns:
            torch.Tensor: the output tensor of the model for this rank's layers
        """
        if self.rank == 0:
            assert input_tensor is not None
            x = input_tensor.to(self.device)
        else:
            x = torch.empty(
                self.part_comm_info[utils.ACTIVATION_SIZE],
                device=self.device,
            )
            dist.recv(x, src=self.rank - 1)

        utils.sync_if_cuda(self.device)
        # if not is_warmup:
        self.batch_times.append(time.perf_counter())

        for layer_idx, layer in enumerate(self.partition):
            layer_input = x.detach().requires_grad_(True)

            if isinstance(layer, nn.ReLU) and layer.inplace:
                layer_output = torch.relu(layer_input)
            else:
                layer_output = layer(layer_input)

            self.activations[microbatch_idx][layer_idx] = (
                layer_input,
                layer_output,
            )
            x = layer_output

        utils.sync_if_cuda(self.device)
        # if not is_warmup:
        self.batch_times.append(time.perf_counter())

        if self.rank < self.world_size - 1:
            dist.send(
                x.detach().contiguous(),
                dst=self.rank + 1,
            )

        return x
        
    def backward(self, model_output: torch.Tensor | None, target: torch.Tensor | None,
                 microbatch_idx: int, is_warmup: bool = False):
        """
        Performs the backward pass of the model, with respect to this module's loss
        function, for this rank.

        Args:
            model_output (torch.Tensor): the output tensor of the full model, 
            if this rank is responsible for the last layers of the model; 
            otherwise None
            target (torch.Tensor): the expected output tensor of the full model,
            if this rank is responsible for the last layers of the model; 
            otherwise None
            microbatch_idx (int): the ID of current microbatch within a given batch
            is_warmup (bool): Whether or not this is a warmup batch
        """
        if self.rank == self.world_size - 1:
            assert model_output is not None
            assert target is not None

            utils.sync_if_cuda(self.device)
            # if not is_warmup:
            self.batch_times.append(time.perf_counter())

            loss = self.loss_fn(model_output, target.to(self.device)) / self.num_microbatches
            grad = torch.autograd.grad(outputs=loss, inputs=model_output,)[0]

        else:
            last_output = self.activations[microbatch_idx][len(self.partition) - 1][1]

            grad = torch.empty(
                self.part_comm_info[utils.GRADIENT_SIZE],
                device=self.device,
                dtype=last_output.dtype,
            )
            dist.recv(grad, src=self.rank + 1)

            utils.sync_if_cuda(self.device)
            # if not is_warmup:
            self.batch_times.append(time.perf_counter())

        for layer_idx in reversed(range(len(self.partition))):
            layer = self.partition[layer_idx]
            layer_input, layer_output = (self.activations[microbatch_idx][layer_idx])

            params = tuple(p for p in layer.parameters() if p.requires_grad)

            grads = torch.autograd.grad(
                outputs=layer_output,
                inputs=(layer_input,) + params,
                grad_outputs=grad,
            )

            grad = grads[0]

            for param, param_grad in zip(params, grads[1:]):
                if param.grad is None:
                    param.grad = param_grad.detach()
                else:
                    param.grad.add_(param_grad.detach())

        utils.sync_if_cuda(self.device)
        # if not is_warmup:
        self.batch_times.append(time.perf_counter())

        if self.rank > 0:
            dist.send(
                grad.detach().contiguous(),
                dst=self.rank - 1,
            )

        self.activations[microbatch_idx].clear()
        
    def train_step(self, input_tensor: torch.Tensor, target: torch.Tensor,
                   is_warmup: bool = False):
        """
        Performs a training step for the model for this rank (i.e. for this 
        rank's layers).

        Args:
            input_tensor (torch.Tensor): the input tensor to the full model
            target (torch.Tensor): the expected output tensor of the full model
        """
        
        self.optimizer.zero_grad(set_to_none=True)

        input_chunks = (
            input_tensor.chunk(self.num_microbatches, dim=0)
            if self.rank == 0 else None
        )
        target_chunks = (
            target.chunk(self.num_microbatches, dim=0)
            if self.rank == self.world_size - 1 else None
        )

        outputs = []
        for microbatch_idx in range(self.num_microbatches):
            output = self.forward(
                input_chunks[microbatch_idx]
                if self.rank == 0 else None,
                microbatch_idx,
                is_warmup=is_warmup,
            )
            outputs.append(output)

        for microbatch_idx in range(self.num_microbatches):
            self.backward(
                outputs[microbatch_idx]
                if self.rank == self.world_size - 1 else None,
                target_chunks[microbatch_idx]
                if self.rank == self.world_size - 1 else None,
                microbatch_idx,
                is_warmup=is_warmup,
            )

        self.optimizer.step()
    
    def eval(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model on the given input.
        
        Args:
            input (torch.Tensor): the input to the model
        Returns:
            torch.Tensor: the output of the model
        """
        
        input_chunks = (
            input.chunk(self.num_microbatches, dim=0)
            if self.rank == 0 else None
        )

        outputs = []
        with torch.no_grad():
            for microbatch_idx in range(self.num_microbatches):
                output = self.forward(
                    input_chunks[microbatch_idx]
                    if self.rank == 0 else None,
                    microbatch_idx,
                    is_warmup=True,
                )
                outputs.append(output.detach())

                self.activations[microbatch_idx].clear()

        return torch.cat(outputs, dim=0)

def train_vgg16_cifar10_pipeline_parallel_worker(
    rank: int, world_size: int, partitions: list[list[nn.Module]],
    communication_info: list[dict[str, torch.Size]], train_loader: DataLoader,
    stats_queue: mp.Queue, num_batches: int = 5, 
    num_warmup_batches: int = 0,
    num_microbatches: int = 2,
    cores_per_rank: int = 1, learning_rate: float = 1e-2, check_output: bool = True
):
    """
    For a given worker process, trains a VGG16 model on the CIFAR-10 dataset 
    using pipeline parallelism.

    Args:
        rank (int): the rank of this process
        world_size (int): the total number of processes
        partitions (list[list[nn.Module]]): the partitions (a list of layers) of 
        the model for each worker
        communication_info (list[dict[str, torch.Size]]): metadata regarding communication 
        between processes, storing the received activation size and gradient 
        size for each partition
        train_loader (DataLoader): the data loader for the training dataset
        stats_queue (mp.Queue): the queue for communicating statistics about this worker
        num_batches (int): the number of batches to train for; defaults to 5
        num_warmup_batches (int): the number of warmup batches where timing
        will be excluded; defaults to 0
        num_microbatches (int): the number of microbatches to split each batch into;
        defaults to 2
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        learning_rate (float): the learning rate of the optimizer; defaults to 1e-2
        check_output (bool): boolean that determines whether the model's output is
        saved; defaults to True
    """
    stats = {
        utils.RANK: rank,
        utils.TOTAL_TIME: 0.0,
        utils.BATCHES_TIMES: []
    }
    
    utils.debug_print(f"Initializing pipeline parallelism on rank {rank}.")
    device = utils.parallel_setup(rank, world_size)
    utils.pin_to_core(rank, cores_per_rank)
    utils.seed_everything(1390)

    pipeline_parallel_wrapper = PipelineParallel(partitions[rank], device, communication_info[rank], 
                                                 stats[utils.BATCHES_TIMES], learning_rate,
                                                 num_microbatches=num_microbatches)

    total_num_batches = num_batches + num_warmup_batches
    
    for i, (inputs, labels) in enumerate(train_loader):
        if i == total_num_batches:
            break

        is_warmup = i < num_warmup_batches

        utils.sync_if_cuda(device)
        step_start = time.perf_counter()

        pipeline_parallel_wrapper.train_step(
            inputs,
            labels,
            is_warmup=is_warmup,
        )

        utils.sync_if_cuda(device)
        step_end = time.perf_counter()

        if not is_warmup:
            stats[utils.TOTAL_TIME] += (step_end - step_start) / num_batches

    if check_output:
        test_output = pipeline_parallel_wrapper.eval(inputs)
        if rank == world_size - 1:
            print("Saving model output")
            torch.save(test_output, utils.SAVED_TENSOR_DIR / f'rank_{rank}_test_output.pt')

    stats_queue.put(stats)
    utils.parallel_cleanup()

def train_vgg16_cifar10_pipeline_parallel(
    world_size: int = 1, num_batches: int = 5, batch_size: int = 32, 
    num_microbatches: int = 2, learning_rate: float = 1e-2, 
    cores_per_rank: int = 1, check_output: bool = True
) -> dict:
    """
    Trains and evaluates a VGG16 model on the CIFAR-10 dataset using 
    pipeline parallelism.
    
    Args:
        world_size (int): the number of processes to train with; defaults to 1
        num_batches (int): the number of batches to train for; defaults to 5
        batch_size (int): the number of data points processed in one step of
        training; defaults to 32
        num_microbatches (int): the number of microbatches to split each batch into;
        defaults to 2
        learning_rate (float): the optimizer's learning rate; defaults to 1e-2
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        check_output (bool): boolean that determines whether the model's output is
        saved;defaults to True
    """
    train_dataset = utils.get_train_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    utils.seed_everything(1390) 
    model = vgg16(weights=None)
    # because of size change for dataset
    model.classifier[0] = nn.Linear(512, 4096)
    model.classifier[6] = nn.Linear(4096, 10)
    
    partitions = utils.split_vgg16(model, world_size)
    assert(len(partitions) == world_size)
    
    assert batch_size % num_microbatches == 0
    microbatch_size = batch_size // num_microbatches
    communication_info = utils.analyze_communication_with_partitions(
        next(iter(train_loader))[0][0:microbatch_size].shape, partitions
    )
    num_warmup_batches = 0
    if torch.cuda.is_available():
        num_warmup_batches = 3
    
    stats_queue = mp.Queue()
    mp.spawn(
        train_vgg16_cifar10_pipeline_parallel_worker,
        args=(world_size, partitions, communication_info, train_loader, 
              stats_queue, num_batches, num_warmup_batches, num_microbatches, cores_per_rank, 
              learning_rate, check_output),
        nprocs=world_size,
        join=True
    )
    return utils.agg_stats_per_rank(stats_queue)
