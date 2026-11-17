# HighPerformanceMLSystems

Distributed VGG16 training with DDP, FSDP, model parallelism, and pipeline
parallelism, plus GPT-2 inference benchmarks for batching and KV caching.
The two subprojects use separate uv environments and require Python 3.12 or later.

## Distributed training

Use Linux with NVIDIA GPUs and a driver compatible with CUDA 12.6. Run these
commands from the repository root. The dataset download is a one-time setup step.

```bash
uv sync --locked --project distributed_training
uv run --locked --project distributed_training python -c 'from torchvision.datasets import CIFAR10; CIFAR10("distributed_training/data", train=True, download=True)'
uv run --locked --project distributed_training python -m distributed_training.src.train ddp --num_workers 2
uv run --locked --project distributed_training python -m distributed_training.benchmarks.plot throughput ddp=2 --output distributed_training/benchmarks/graphs/throughput.png
```

Request at least one GPU per worker and enough CPUs for
`--num_workers * --cores_per_worker`. To submit a profiling sweep on a Slurm
cluster with the configured `gpus` partition and GPU constraints, run
`uv run --locked --project distributed_training python -m distributed_training.benchmarks.profiler ddp`.
Training statistics are written to `distributed_training/data/` and Slurm logs
to `distributed_training/benchmarks/profile_logs/`.

## Inference serving

```bash
cd inference_serving
uv sync --locked
uv run --locked scripts/run.py serve --use_kv
uv run --locked scripts/run.py plot --min_sustained 0
```

The first inference run needs network access for the GPT-2 tokenizer and model
weights unless they are already cached. Serving defaults to
`benchmarks/traces/short_uniform.jsonl`. The offline `kvcache` and
`speculative` commands use `long_uniform.jsonl`. Commands write statistics to
`stat/` and plots to `graph/` in the inference subproject by default.

On Slurm, use `./scripts/run.sh serve --use_kv` from the inference subproject.
The runnable serving backend is static batching. Continuous batching, paged
batching, and `plot_memory` are implementation placeholders.
