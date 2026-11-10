"""Small, offline DataLoader experiments. Run this file to start workers safely."""

import argparse
import gc
import json
import multiprocessing as mp
import os
import random
from statistics import median
from time import perf_counter, sleep

# Set these before importing libraries when this file runs as a program.
# Spawned workers inherit them. Importing this module leaves the caller's
# thread settings alone.
if __name__ == "__main__":
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset, get_worker_info
except ModuleNotFoundError as error:
    if error.name not in {"torch", "numpy"}:
        raise
    print("Optional lesson dependencies are missing. Install:")
    print("python -m pip install -r requirements-advanced.txt -r requirements-torch.txt")
    print('RESULT_JSON={"status": "skipped", "reason": "missing dependency"}')
    raise SystemExit(0)


# SECTION: dataset
class TinySamples(Dataset):
    """Create each sample when requested; keep no large list in worker memory."""

    def __init__(self, count=64, delay=0.004, random_draws=False):
        self.count = count
        self.delay = delay
        self.random_draws = random_draws

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        sleep(self.delay)  # Stand-in for loading/preparation; not real image decoding.
        sample = {
            "image": torch.full((3, 16, 16), index / self.count, dtype=torch.float32),
            "index": index,
        }
        if self.random_draws:
            worker = get_worker_info()
            sample.update(
                worker=-1 if worker is None else worker.id,
                python_draw=random.random(),
                numpy_draw=float(np.random.random()),
                torch_draw=float(torch.rand(())),
            )
        return sample
# END SECTION: dataset


# SECTION: seeds
def seed_worker(worker_id):
    """Initialize each library's sequence of random numbers for this worker."""
    # PyTorch has already set this worker's seed. NumPy accepts a 32-bit seed.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.set_num_threads(1)


def make_loader(dataset, workers, batch_size, prefetch=1, persistent=False,
                seed=123, shuffle=False, pin_memory=False):
    generator = torch.Generator().manual_seed(seed)
    options = dict(
        batch_size=batch_size,
        num_workers=workers,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
        pin_memory=pin_memory,
    )
    if workers:
        options.update(
            multiprocessing_context="spawn",
            worker_init_fn=seed_worker,
            prefetch_factor=prefetch,
            persistent_workers=persistent,
            timeout=60,
        )
    else:
        # With no worker processes, worker_init_fn would not run.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return DataLoader(dataset, **options)
# END SECTION: seeds


# SECTION: timing
def read_epoch(loader, sample_count, consumer_delay):
    """Time a complete pass and check that every index was returned once."""
    start = perf_counter()
    iterator = iter(loader)
    indices, waits = [], []
    first_batch_seconds = None
    try:
        while True:
            wait_start = perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            waits.append(perf_counter() - wait_start)
            if first_batch_seconds is None:
                first_batch_seconds = perf_counter() - start
            indices.extend(batch["index"].tolist())
            # Same amount of simulated consumer work per sample in every trial.
            sleep(consumer_delay * len(batch["index"]))
    finally:
        del iterator
    seconds = perf_counter() - start
    assert sorted(indices) == list(range(sample_count)), "Missing or duplicate sample"
    return {
        "seconds": seconds,
        "samples_per_second": sample_count / seconds,
        "first_batch_seconds": first_batch_seconds,
        "median_later_wait_ms": 1000 * median(waits[1:]) if len(waits) > 1 else 0,
        "samples": len(indices),
    }


def benchmark_configuration(config, count=64, delay=0.004, consumer_delay=0.002):
    dataset = TinySamples(count=count, delay=delay)
    loader = make_loader(dataset, **config)
    try:
        epochs = [read_epoch(loader, count, consumer_delay) for _ in range(2)]
    finally:
        # Persistent workers outlive an epoch. Drop the loader after full passes.
        del loader
        gc.collect()
    sample_bytes = 3 * 16 * 16 * 4
    scheduled_batches = config["workers"] * config["prefetch"]
    return {
        "config": config,
        "epochs": epochs,
        "prefetch_tensor_payload_mib": (
            scheduled_batches * config["batch_size"] * sample_bytes / 2**20
        ),
    }


def run_benchmarks(count, delay, consumer_delay):
    # Adjacent comparisons each change one setting; use identical generated data.
    configurations = [
        dict(workers=0, batch_size=4, prefetch=1, persistent=False),
        dict(workers=2, batch_size=4, prefetch=1, persistent=False),
        dict(workers=2, batch_size=4, prefetch=1, persistent=True),
        dict(workers=2, batch_size=4, prefetch=2, persistent=True),
        dict(workers=2, batch_size=8, prefetch=2, persistent=True),
    ]
    results = []
    for config in configurations:
        print("Running:", config, flush=True)
        results.append(benchmark_configuration(config, count, delay, consumer_delay))
    return results
# END SECTION: timing


# SECTION: randomness
def collect_random_epochs(seed=123):
    loader = make_loader(
        TinySamples(count=12, delay=0, random_draws=True),
        workers=2, batch_size=3, persistent=True, shuffle=True, seed=seed,
    )
    epochs = []
    try:
        for _ in range(2):
            rows = []
            for batch in loader:
                for position in range(len(batch["index"])):
                    rows.append({
                        key: batch[key][position].item()
                        for key in ("index", "worker", "python_draw", "numpy_draw", "torch_draw")
                    })
            assert sorted(row["index"] for row in rows) == list(range(12))
            epochs.append(rows)
    finally:
        del loader
        gc.collect()
    return epochs


def run_randomness():
    first_run = collect_random_epochs(seed=123)
    repeated_run = collect_random_epochs(seed=123)
    assert first_run == repeated_run, "Same setup and seed should repeat this example"
    by_index = [{row["index"]: row for row in epoch} for epoch in first_run]
    changed_draws = any(
        by_index[0][index]["python_draw"] != by_index[1][index]["python_draw"]
        for index in range(12)
    )
    assert changed_draws
    return {
        "same_two_epoch_runs": first_run == repeated_run,
        "different_epoch_draws": changed_draws,
        "epoch_orders": [[row["index"] for row in epoch] for epoch in first_run],
        "first_three_rows_per_epoch": [epoch[:3] for epoch in first_run],
    }
# END SECTION: randomness


# SECTION: profiling
def stage_profile(device="cpu", pin_memory=False):
    """Use barriers to separate stages. This intentionally removes GPU overlap."""
    if device == "cuda" and not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA is not available"}

    target = torch.device(device)
    # Make weights for a tiny matrix multiplication and warm up the device.
    # inference_mode skips recording information used to calculate gradients.
    weights = torch.linspace(0, 1, 3 * 16 * 16 * 16, device=target).reshape(-1, 16)
    with torch.inference_mode():
        warmup = torch.zeros((4, 3 * 16 * 16), device=target) @ weights
    if device == "cuda":
        torch.cuda.synchronize()
    del warmup

    loader = make_loader(
        TinySamples(count=32, delay=0.004), workers=2, batch_size=4,
        persistent=True, pin_memory=pin_memory,
    )
    rows = []
    try:
        # Finish one warm-up pass so worker startup is outside the stage totals.
        for batch in loader:
            pass
        iterator = iter(loader)
        indices = []
        try:
            with torch.inference_mode():
                while True:
                    start = perf_counter()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    wait = perf_counter() - start
                    indices.extend(batch["index"].tolist())

                    start = perf_counter()
                    images = batch["image"].to(target, non_blocking=(device == "cuda"))
                    if device == "cuda":
                        torch.cuda.synchronize()
                    transfer = perf_counter() - start

                    start = perf_counter()
                    output = images.flatten(1) @ weights
                    if device == "cuda":
                        torch.cuda.synchronize()
                    compute = perf_counter() - start
                    rows.append(dict(wait=wait, transfer=transfer, compute=compute))
                    # This use happens outside timing, after CUDA synchronization.
                    assert bool(torch.isfinite(output).all())
            assert sorted(indices) == list(range(32))
        finally:
            del iterator
    finally:
        del loader
        gc.collect()
    return {
        "status": "ok", "device": device, "pin_memory": pin_memory,
        "batches": len(rows),
        "totals_ms": {key: 1000 * sum(row[key] for row in rows) for key in rows[0]},
    }
# END SECTION: profiling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "single", "randomness", "profile", "all"), default="all")
    parser.add_argument("--workers", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prefetch", type=int, choices=(1, 2), default=1)
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--delay", type=float, default=0.004)
    parser.add_argument("--consumer-delay", type=float, default=0.002)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pin-memory", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1 or min(args.delay, args.consumer_delay) < 0:
        parser.error("count and batch-size must be positive; delays must be nonnegative")
    if args.persistent and not args.workers:
        parser.error("persistent workers require workers > 0")
    torch.set_num_threads(1)
    result = {"status": "ok", "torch_version": torch.__version__}
    if args.mode in ("benchmark", "all"):
        result["benchmark"] = run_benchmarks(args.count, args.delay, args.consumer_delay)
    if args.mode == "single":
        config = dict(workers=args.workers, batch_size=args.batch_size,
                      prefetch=args.prefetch, persistent=args.persistent)
        result["benchmark"] = [benchmark_configuration(config, args.count, args.delay, args.consumer_delay)]
    if args.mode in ("randomness", "all"):
        result["randomness"] = run_randomness()
    if args.mode in ("profile", "all"):
        result["profile"] = stage_profile(args.device, args.pin_memory)
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    mp.freeze_support()
    main()
