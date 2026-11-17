import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torchvision.models import vgg16
import json

from . import utils

class FullyShardedDataParallel():
    """
    A module implementation of Fully Sharded Data Parallel (FSDP)

    Attributes:
        - layers (nn.ModuleList): a list of all of the layers of the model
        - local_params (list[list[nn.Parameter]]): a list containing, for each 
          layer, either an empty list (if the layer has no parameters) or a list 
          containing this rank's shard of the layer's weight and bias parameters
        - device (torch.device): the device the parts of the model for this
          rank are to be located on
        - rank (int): the rank of the process the module is located on
        - world_size (int): the total number of processes running FSDP
        - learning_rate (float): the learning rate used for training FSDP
        - optimizer (torch.optim.SGD): the optimizer for only the tensor shards 
          on this worker
        - loss_fn (nn.Module): the loss function used when training this model
    """
    def __init__(self, layers: nn.ModuleList, device: torch.device, 
                 unsharded_param_tensors: list[list[tuple[torch.Tensor, torch.Size]]], 
                 memory_log: list,
                 learning_rate: float = 0.001, loss_fn: nn.Module = nn.CrossEntropyLoss()):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.layers = layers
        self.unsharded_param_shapes = [[layer_param_data[1] for layer_param_data in layer_params] 
                                       for layer_params in unsharded_param_tensors]
        self.device = device
        self.local_params = self.get_local_info(unsharded_param_tensors)
        self.learning_rate = learning_rate
        self.activations: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.optimizer = torch.optim.SGD(
            [param for param_list in self.local_params for param in param_list],
            lr=self.learning_rate
        )
        self.loss_fn = loss_fn
        self.memory_log = memory_log

    def get_local_info(
        self, unsharded_param_tensors: list[list[tuple[torch.Tensor, torch.Size]]]
    ) -> list[list[nn.Parameter]]:
        """
        Shards the given flattened and padded tensors for all parameters in a model, 
        and returning just the local parameter shards for this rank.
        
        Args:
            unsharded_param_tensors (list[list[tuple[torch.Tensor, torch.Size]]]): a 
            list containing, for each layer, either an empty list (if the layer has no 
            parameters) or a list containing the layer's weight and bias *flattened* 
            parameters, each padded to be a multiple of `world_size`, tupled with 
            their *unflattened* sizes
        Returns:
            list[list[nn.Parameter]]: a list containing, for each layer, either
            an empty list (if the layer has no parameters) or a list containing
            this rank's shard of the layer's weight and bias parameters
        """
        local_params = []

        for layer_params in unsharded_param_tensors:
            local_layer_params = []

            for flat_param, original_shape in layer_params:
                shard_size = flat_param.numel() // self.world_size
                start = self.rank * shard_size
                shard = (
                    flat_param[start:start + shard_size]
                    .detach()
                    .clone()
                    .to(self.device)
                )
                local_layer_params.append(nn.Parameter(shard))
            local_params.append(local_layer_params)

        return local_params

    def gather_param_data(self, layer_idx: int) -> list[torch.Tensor]:
        """
        Consumes a layer index, returns the full, unsharded, unflattened
        parameter tensor for that layer
        """
        
        full_params = []

        with torch.no_grad():
            for param_idx, local_param in enumerate(
                self.local_params[layer_idx]
            ):
                shards = [
                    torch.empty_like(local_param)
                    for _ in range(self.world_size)
                ]

                dist.all_gather(
                    shards,
                    local_param.detach().contiguous(),
                )

                flat_param = torch.cat(shards, dim=0)
                original_shape = self.unsharded_param_shapes[
                    layer_idx
                ][param_idx]

                full_params.append(
                    flat_param[:original_shape.numel()].reshape(
                        original_shape
                    )
                )

                del shards, flat_param

        return full_params

    def record_memory(self, layer_idx, phase):
        if (
            self.device.type == "cuda"
            and getattr(self, "log_memory_enabled", False)
        ):
            self.memory_log.append(
                utils.log_memory(
                    self.device,
                    self.rank,
                    batch_idx,
                    layer_idx,
                    phase,
                )
            )

    def forward(self, input: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Performs a forward pass on the underlying model.
        
        Args:
            input (torch.Tensor): the input to the model
            batch_idx (int): the index of the current batch
        Returns:
            torch.Tensor: the output of the model
        """
        
        # TODO: Implement!
        self.activations.clear()
        x = input.to(self.device)

        with torch.no_grad():
            for layer_idx, layer in enumerate(self.layers):
                layer.eval()
                self.record_memory(layer_idx, "forward_before_gather")

                full_params = self.gather_param_data(layer_idx)

                if full_params:
                    layer.weight = nn.Parameter(
                        full_params[0], requires_grad=False
                    )
                    layer.bias = nn.Parameter(
                        full_params[1], requires_grad=False
                    )

                self.record_memory(layer_idx, "forward_after_gather")

                layer_input = x

                if isinstance(layer, nn.ReLU) and layer.inplace:
                    layer_output = torch.relu(layer_input)
                else:
                    layer_output = layer(layer_input)

                self.activations[layer_idx] = (
                    layer_input,
                    layer_output,
                )
                x = layer_output

                self.record_memory(layer_idx, "forward_after_compute")

                if full_params:
                    layer.weight = None
                    layer.bias = None

                del full_params

                self.record_memory(layer_idx, "forward_after_release")

        return x.detach().requires_grad_(True)

    def backward(self, model_output: torch.Tensor, target: torch.Tensor,
                 batch_idx: int) -> torch.Tensor:
        """
        Performs a backward pass on the underlying model.
        
        Args:
            model_output (torch.Tensor): the output of the model
            target (torch.Tensor): the expected output of the model
            batch_idx (int): the index of the current batch
        Returns:
            torch.Tensor: the calculated loss of the model's output compared
            to the expected output
        """
        
        loss = self.loss_fn(model_output, target.to(self.device))
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=model_output,
        )[0]

        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]

            self.record_memory(layer_idx, "backward_before_gather")

            full_params = self.gather_param_data(layer_idx)

            if full_params:
                layer.weight = nn.Parameter(full_params[0])
                layer.bias = nn.Parameter(full_params[1])
                params = (layer.weight, layer.bias)
            else:
                params = ()

            self.record_memory(layer_idx, "backward_after_gather")

            layer_input = (
                self.activations[layer_idx][0]
                .detach()
                .requires_grad_(True)
            )

            if isinstance(layer, nn.ReLU) and layer.inplace:
                layer_output = torch.relu(layer_input)
            else:
                layer_output = layer(layer_input)

            grads = torch.autograd.grad(
                outputs=layer_output,
                inputs=(layer_input,) + params,
                grad_outputs=grad,
            )

            grad = grads[0].detach()

            self.record_memory(layer_idx, "backward_after_compute")

            for param_idx, local_param in enumerate(
                self.local_params[layer_idx]
            ):
                flat_grad = grads[param_idx + 1].detach().reshape(-1)

                padded_grad = torch.zeros(
                    local_param.numel() * self.world_size,
                    dtype=flat_grad.dtype,
                    device=self.device,
                )
                padded_grad[:flat_grad.numel()].copy_(flat_grad)

                if self.device.type == "cuda":
                    local_grad = torch.empty_like(local_param)

                    dist.reduce_scatter_tensor(
                        local_grad,
                        padded_grad,
                        op=dist.ReduceOp.SUM,
                    )
                else:
                    dist.all_reduce(
                        padded_grad,
                        op=dist.ReduceOp.SUM,
                    )

                    start = self.rank * local_param.numel()
                    local_grad = padded_grad[
                        start:start + local_param.numel()
                    ].clone()

                local_grad.div_(self.world_size)

                if local_param.grad is None:
                    local_param.grad = local_grad
                else:
                    local_param.grad.add_(local_grad)

                del flat_grad, padded_grad, local_grad

            self.record_memory(layer_idx, "backward_after_gradient_sync")

            if full_params:
                layer.weight = None
                layer.bias = None

            del params, full_params, grads
            del layer_input, layer_output
            del self.activations[layer_idx]

            self.record_memory(layer_idx, "backward_after_release")

        return loss.detach()

    def optimizer_step(self):
        """
        Updates the parameters of the model using the optimizer and accumulated
        gradients, and then resets the optimizer/accumulated gradients for the
        next training step. Should be called after each batch's backward pass 
        completes.
        """
        
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)


def train_vgg16_cifar10_fsdp_worker(
    rank: int, world_size: int, train_dataset: Dataset, stats_queue: mp.Queue,
    layers: nn.ModuleList, unsharded_param_tensors: list[list[tuple[torch.Tensor, torch.Size]]],
    num_warmup_batches: int = 0,
    num_batches: int = 5, cores_per_rank: int = 1, batch_size = 32, 
    learning_rate: float = 1e-2, check_weights: bool = False, check_output: bool = True
):
    """
    For a given worker process, trains a VGG16 model on the CIFAR-10 
    dataset using FSDP.

    Args:
        rank (int): the rank of the current process
        world_size (int): the total number of processes
        train_dataset (Dataset): the training dataset
        stats_queue (mp.Queue): the queue for communicating statistics about this worker
        layers (nn.ModuleList): a list of layers in the model to be trained
        unsharded_param_tensors (list[list[tuple[torch.Tensor, torch.Size]]]): a 
        list containing, for each layer, either an empty list (if the layer has no 
        parameters) or a list containing the layer's weight and bias *flattened* 
        parameters, each padded to be a multiple of `world_size`, tupled with 
        their *unflattened* sizes
        num_batches (int): the number of batches to train for; defaults to 5
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        batch_size (int): the number of data points processed in one step of
        training; defaults to 64
        learning_rate (float): the learning rate of the optimizer; defaults to 1e-2
        check_weights (bool): boolean to determine whether weights are being saved;
        defaults to False
        check_output (bool): boolean to determine whether model output on a single
        batch is saved; defaults to True
    """
    stats = {
        utils.RANK: rank,
        utils.TOTAL_TIME: 0.0,
    }

    utils.debug_print(f"Initializing FSDP on rank {rank}.")
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

    memory_log = []
    model = FullyShardedDataParallel(layers, device, unsharded_param_tensors, memory_log, learning_rate)
  
    total_batches = num_warmup_batches + num_batches

    if len(train_loader) < total_batches + int(check_output):
        raise ValueError("Not enough batches for this run")

    if device.type == "cuda":
        utils.sync_if_cuda(device)
        torch.cuda.reset_peak_memory_stats(device)

    for i, (inputs, labels) in enumerate(train_loader):
        if i == total_batches:
            break

        is_warmup = i < num_warmup_batches
        model.log_memory_enabled = is_warmup

        utils.sync_if_cuda(device)
        step_start = time.perf_counter()

        inputs = inputs.to(device)
        labels = labels.to(device)

        if device.type == "cuda" and is_warmup:
            memory_log.append(
                utils.log_memory(
                    device, rank, i, -1, "step_start"
                )
            )

        outputs = model.forward(inputs, i)
        loss = model.backward(outputs, labels, i)
        model.optimizer_step()

        if device.type == "cuda" and is_warmup:
            memory_log.append(
                utils.log_memory(
                    device, rank, i, -1, "after_optimizer_step"
                )
            )

        del outputs, loss

        utils.sync_if_cuda(device)
        step_end = time.perf_counter()

        if not is_warmup:
            stats[utils.TOTAL_TIME] += step_end - step_start

    model.log_memory_enabled = False

    if check_weights:
        state_dict = {}
        for i, layer in enumerate(model.layers):
            full_tensors = model.gather_param_data(i)
            if full_tensors:
                state_dict[f'{i}.weight'] = full_tensors[0].cpu()
                state_dict[f'{i}.bias'] = full_tensors[1].cpu()
        torch.save(state_dict, utils.STATE_DICT_DIR / f'rank_{rank}_weights.pt')
    if check_output:
        print(f"Saving model output for rank {rank}")
        inputs = inputs.to(device)
        torch.save(model.forward(inputs, 0).cpu(), utils.SAVED_TENSOR_DIR / f'rank_{rank}_test_output.pt')

    if len(memory_log) > 0:
        utils.MEMORY_LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = utils.MEMORY_LOG_DIR / f'fsdp{world_size}_rank{rank}_memory_log.pt'
        with path.open('w') as f:
            json.dump(memory_log, f)


    if rank == 0:
        print("Finished training")

    stats_queue.put(stats)
    utils.parallel_cleanup()


def init_layers_and_params(world_size: int) -> tuple[nn.ModuleList, list[list[tuple[torch.Tensor, torch.Size]]]]:
    """
    Loads/initializes the VGG16 model and returns its layers and each layer's 
    padded parameter tensors.
    
    Args:
        world_size (int): the number of worker processes running FSDP
    Returns:
        tuple[nn.ModuleList, list[list[tuple[torch.Tensor, torch.Size]]]: a tuple of:
         - the list of layers in the VGG16 model
         - a list containing, for each layer, either an empty list (if the layer 
           has no parameters) or a list containing the layer's weight and bias 
           *flattened* parameters, each padded to be a multiple of `world_size`, 
           tupled with their *unflattened* sizes
    """
    utils.seed_everything(1390)
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 10)
    model.eval()
    
    def pad_and_flatten_param(orig_param_tensor: torch.Tensor) -> torch.Tensor:
        orig_tensor_size = orig_param_tensor.numel()
        if orig_tensor_size % world_size != 0:
            padded_tensor_size = orig_tensor_size + (world_size - (orig_tensor_size % world_size))
            padded_param_tensor = torch.zeros(size=(padded_tensor_size,))
            padded_param_tensor[0:orig_tensor_size] = orig_param_tensor.flatten()
            return padded_param_tensor
        else:
            return orig_param_tensor.flatten()

    layers = nn.ModuleList([layer for layer in model.features] + [model.avgpool, nn.Flatten()] + [layer for layer in model.classifier])
    unsharded_param_tensors: list[list[tuple[torch.Tensor, torch.Size]]] = []  
    for layer in layers:
        if hasattr(layer, 'weight'):
            unsharded_param_tensors.append([
                (pad_and_flatten_param(layer.weight.data), layer.weight.data.shape), 
                (pad_and_flatten_param(layer.bias.data), layer.bias.data.shape)
            ])
        else:
            unsharded_param_tensors.append([])
                
    return layers, unsharded_param_tensors


def train_vgg16_cifar10_fsdp(
    world_size: int = 1, num_batches: int = 5, batch_size: int = 32,
    learning_rate: float = 1e-2, cores_per_rank = 1, 
    check_weights: bool = False, check_output: bool = True
) -> dict:
    """
    Trains a VGG16 model on the CIFAR-10 dataset using FSDP.
    
    Args:
        world_size (int): the number of processes to train with; defaults to 1
        num_batches (int): the number of batches to train for; defaults to 5
        batch_size (int): the number of data points processed in one step of
        training; defaults to 32
        learning_rate (float): the optimizer's learning rate; defaults to 1e-2
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        check_weights (bool): boolean to determine whether weights are being saved;
        defaults to False
        check_output (bool): boolean to determine whether model output on a single
        batch is saved; defaults to True
    """    
    layers, unsharded_param_tensors = init_layers_and_params(world_size)
    utils.debug_print("Created model parameter information")
    
    stats_queue = mp.Queue()
    num_warmup_batches = 0
    if torch.cuda.is_available():
        num_warmup_batches = 3
    mp.spawn(
        train_vgg16_cifar10_fsdp_worker,
        args=(world_size, utils.get_train_dataset(), stats_queue, layers, 
              unsharded_param_tensors, 
              num_warmup_batches,
              num_batches, cores_per_rank, batch_size, 
              learning_rate, check_weights, check_output),
        nprocs=world_size,
        join=True
    )

    return utils.agg_stats_per_rank(stats_queue)
