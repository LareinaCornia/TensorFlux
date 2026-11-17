"""
Plots of the per-run stats saved by the client (see `client.save_stats`).

You will implement `plot_throughput_latency`, which draws throughput against latency
(median by default; p99 / p90 / mean also available), with one line per
(server type, KV vs baseline, batch size, block size for paged runs).

By default, it drops a point where the server can't keep up with the offered
load if its achieved throughput is below 90% of the offered load
(min_sustained, set to 0 to plot the full saturated tail).
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

# Metric key in each stats file -> human-readable axis label.
METRIC_LABELS = {
    "median_latency": "median (p50) E2E latency (s)",
    "p90_latency": "p90 E2E latency (s)",
    "p99_latency": "p99 E2E latency (s)",
    "mean_latency": "mean E2E latency (s)",
}
METRIC_KEYS = {"median": "median_latency", "p90": "p90_latency",
               "p99": "p99_latency", "mean": "mean_latency"}
X_LABELS = {
    "rate": "offered load (req/s)",
    "achieved_throughput": "achieved throughput (req/s)",
    "effective_rate": "effective submission rate (req/s)",
}


def load_records(stat_dir: str) -> list[dict]:
    records = []
    for fp in sorted(glob.glob(os.path.join(stat_dir, "*.json"))):
        try:
            with open(fp) as f:
                records.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[plot] skipping {fp}: {e}")
    return records


def save_figure(out: str):
    out_dir = os.path.dirname(out) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def trace_name(record: dict) -> str:
    """Short name of the trace a run replayed, e.g. 'short_uniform'."""
    trace_file = record.get("trace_file")
    return os.path.splitext(os.path.basename(trace_file))[0] if trace_file else "trace"


def plot_throughput_latency(stat_dir: str = "stat", out: str = "graph/throughput_latency.png",
                            metric: str = "median", x: str = "achieved_throughput",
                            server_type: str | None = None, min_sustained: float = 0.9,
                            rates: str | None = None, trace: str | None = None):
    """
    Plots latency against throughput for one trace, one line per
    (server type, KV vs baseline, batch size, block size for paged runs).

    Args:
        stat_dir: directory of client stats JSON files
        out: output image path
        metric: latency statistic for the y-axis ('median', 'p90', 'p99' or 'mean')
        x: x-axis quantity ('rate', 'achieved_throughput' or 'effective_rate')
        server_type: only plot runs with this server_type (default: all)
        min_sustained: drop points whose achieved_throughput < this fraction of the
        offered load (rate); set 0 to keep the saturated tail
        rates: comma-separated offered loads to include (default: every rate present)
        trace: only plot runs of this trace, by name (e.g. 'short_uniform') or path;
        required when `stat_dir` holds runs of more than one trace
    """
    if metric not in METRIC_KEYS:
        raise ValueError(f"unknown metric: {metric}")
    if x not in X_LABELS:
        raise ValueError(f"unknown x-axis: {x}")
    if not 0 <= min_sustained <= 1:
        raise ValueError("min_sustained must be between 0 and 1")

    records = load_records(stat_dir)
    if not records:
        raise ValueError(f"no JSON stats files found in {stat_dir}")

    if trace is not None:
        selected_trace = os.path.splitext(os.path.basename(trace))[0]
        records = [record for record in records if trace_name(record) == selected_trace]
    else:
        traces = {trace_name(record) for record in records}
        if len(traces) != 1:
            raise ValueError(
                "stat_dir contains multiple traces; choose one with --trace: "
                + ", ".join(sorted(traces))
            )
        selected_trace = next(iter(traces))

    if server_type is not None:
        records = [
            record for record in records
            if record.get("server_type") == server_type
        ]
    if rates is not None:
        allowed_rates = {
            float(value.strip()) for value in rates.split(",") if value.strip()
        }
        records = [
            record for record in records
            if float(record.get("rate")) in allowed_rates
        ]

    metric_key = METRIC_KEYS[metric]
    required = {
        "server_type", "use_kv", "batch_size", "rate",
        "achieved_throughput", x, metric_key,
    }
    records = [record for record in records if required.issubset(record)]
    records = [
        record for record in records
        if float(record["achieved_throughput"])
        >= min_sustained * float(record["rate"])
    ]
    if not records:
        raise ValueError("no stats points remain after applying the requested filters")

    grouped = defaultdict(list)
    for record in records:
        kind = record["server_type"]
        block_size = record.get("block_size") if kind == "paged" else None
        key = (
            kind,
            bool(record["use_kv"]),
            int(record["batch_size"]),
            block_size,
        )
        grouped[key].append(record)

    plt.figure(figsize=(7, 5))
    for (kind, use_kv, batch_size, block_size), points in sorted(grouped.items()):
        points.sort(key=lambda record: float(record[x]))
        label = f"{kind}, {'KV cache' if use_kv else 'No KV cache'}, batch={batch_size}"
        if block_size is not None:
            label += f", block={block_size}"
        plt.plot(
            [float(record[x]) for record in points],
            [float(record[metric_key]) for record in points],
            marker="o",
            linestyle="-" if use_kv else "--",
            label=label,
        )

    plt.xlabel(X_LABELS[x])
    plt.ylabel(METRIC_LABELS[metric_key])
    plt.title(f"Throughput vs. latency — {selected_trace}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(out)


def plot_memory(stat_dir: str = "stat", out: str = "graph/kv_memory.png",
                trace: str | None = None, rates: str | None = None):
    """
    Plots KV cache memory utilization (%) over time of continuous (slot) and paged runs
    for one trace, one line per run.

    Args:
        stat_dir: directory of client stats JSON files
        out: output image path
        trace: only plot runs of this trace, by name (e.g. 'mixed_short_long') or path;
        required when `stat_dir` holds runs of more than one trace
        rates: comma-separated offered loads to include (default: every rate present)
    """
    # TODO (Part 3B): Plot Graph 6 of the handout and save it to `out`.
    #   Each continuous or paged record has a `memory_log`: one sample per scheduler step,
    #   each a dict with 'time', 'cached' (tokens actually in the KV cache) and 'reserved'
    #   (tokens set aside for the running requests).
    #   x-axis: time (s) since the run's first sample.
    #   y-axis: memory utilization (%) = 100 * cached / total KV cache tokens, where the
    #   total is the record's `kv_cache_tokens` (batch_size * max_seq_len if it is None).
    #   Draw one line per run (server type, and block size for paged runs), comparing slots
    #   and blocks at the same KV cache size, same trace and same rate.
    raise NotImplementedError("plot_memory")


# Add plotting functions below (use `load_records` and `save_figure`),
# and call them from `run.py`.
