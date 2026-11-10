"""Tiny, offline DataLoader examples. Run this file to start worker processes."""

import argparse
from collections import Counter
import json
import multiprocessing as mp
import os

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    print("PyTorch is optional. Install it with: python -m pip install -r requirements-torch.txt")
    raise SystemExit(0)


class TinyImages(Dataset):
    """A map-style dataset: request one sample using its index."""

    def __init__(self, count=8):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        worker = get_worker_info()
        # A tiny generated image: 1 color channel, 2 rows, 2 columns.
        image = torch.full((1, 2, 2), float(index)) / max(self.count - 1, 1)
        return {
            "image": image,
            "index": index,
            "worker": -1 if worker is None else worker.id,
            "pid": os.getpid(),
        }


class NumberStream(IterableDataset):
    """A stream, with an intentional duplicate-data option for teaching."""

    def __init__(self, count=8, shard=True):
        self.count = count
        self.shard = shard

    def __iter__(self):
        worker = get_worker_info()
        start, step = 0, 1
        if self.shard and worker is not None:
            # Worker 0 takes 0, 2, 4, 6; worker 1 takes 1, 3, 5, 7.
            start, step = worker.id, worker.num_workers
        for number in range(start, self.count, step):
            yield {
                "index": number,
                "worker": -1 if worker is None else worker.id,
                "pid": os.getpid(),
            }


def collect_rows(loader):
    """Read every batch so each worker can finish and shut down normally."""
    rows = []
    for batch in loader:
        print(
            "indices:", batch["index"].tolist(),
            "workers:", batch["worker"].tolist(),
            "process IDs:", batch["pid"].tolist(),
        )
        rows.extend(
            {"index": int(index), "worker": int(worker), "pid": int(pid)}
            for index, worker, pid in zip(batch["index"], batch["worker"], batch["pid"])
        )
    return rows


def run_map_demo():
    dataset = TinyImages(count=8)
    results = {}
    # The main process chooses these two index orders using the sampler.
    for label, order in (("forward", list(range(8))), ("reverse", list(reversed(range(8))))):
        loader = DataLoader(
            dataset,
            batch_size=2,
            sampler=order,
            num_workers=2,
            multiprocessing_context="spawn",
            prefetch_factor=1,
        )
        print(f"\nMap-style order: {label}; main process ID: {os.getpid()}")
        rows = collect_rows(loader)
        assert [row["index"] for row in rows] == order
        assert {row["worker"] for row in rows} == {0, 1}
        assert all(row["pid"] != os.getpid() for row in rows)
        results[label] = rows
    return results


def run_stream_demo():
    results = {}
    for shard in (False, True):
        loader = DataLoader(
            NumberStream(count=8, shard=shard),
            batch_size=2,
            num_workers=2,
            multiprocessing_context="spawn",
            prefetch_factor=1,
        )
        label = "sharded" if shard else "unsharded"
        print(f"\nIterable-style stream: {label}")
        rows = collect_rows(loader)
        counts = Counter(row["index"] for row in rows)
        expected_copies = 1 if shard else 2
        assert counts == Counter({index: expected_copies for index in range(8)})
        print("Occurrences per index:", dict(sorted(counts.items())))
        results[label] = rows
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("map", "stream", "all"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = {}
    if args.mode in ("map", "all"):
        results["map"] = run_map_demo()
    if args.mode in ("stream", "all"):
        results["stream"] = run_stream_demo()
    # A compact final line also lets automated checks inspect the real results.
    print("RESULT_JSON=" + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    mp.freeze_support()
    main()
