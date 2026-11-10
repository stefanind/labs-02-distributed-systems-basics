"""Small bounded file -> thread -> process -> consumer pipeline for notebook 12."""

import argparse
import asyncio
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zlib import crc32


# Every generated file starts with these identifying bytes (its header).
MAGIC = b"MLDEMO1\n"


def make_inputs(folder, jobs, byte_count):
    """Create actual local files; only path metadata is kept in the input list."""
    items = []
    for job_id in range(jobs):
        pattern = bytes((value + job_id) % 256 for value in range(256))
        payload = (pattern * ((byte_count + 255) // 256))[:byte_count]
        path = folder / f"sample_{job_id:03d}.bin"
        path.write_bytes(MAGIC + payload)
        items.append((job_id, path))
    return items


def read_file(item, read_delay):
    job_id, path = item
    data = path.read_bytes()  # Actual blocking file I/O occurs in a thread.
    if read_delay:
        time.sleep(read_delay)  # Optional, explicit simulation of slower storage.
    return job_id, data


def transform(item, rounds):
    """Synthetic byte processing in Python. This is not an image decoder."""
    job_id, data = item
    if not data.startswith(MAGIC):
        raise ValueError(f"Corrupt input {job_id}: missing sample header")
    data = data[len(MAGIC):]
    for _ in range(rounds):
        data = bytes((value * 3 + job_id) % 256 for value in data)
    return job_id, data


def sequential_baseline(items, rounds, read_delay, batch_size, consumer_delay):
    seen = {}
    batch = []
    for item in items:
        batch.append(transform(read_file(item, read_delay), rounds))
        if len(batch) == batch_size:
            time.sleep(consumer_delay)
            seen.update((job_id, crc32(data)) for job_id, data in batch)
            batch.clear()
    if batch:
        time.sleep(consumer_delay)
        seen.update((job_id, crc32(data)) for job_id, data in batch)
    return seen


async def coordinate(inputs, results, pool, args, stats):
    """One coordinator owns at most one job as it moves through the stages."""
    # args holds command-line settings; stats holds observed activity counts.
    loop = asyncio.get_running_loop()
    while True:
        # No await during this claim: other tasks on this event loop cannot
        # interrupt next(). The iterator contains paths, not loaded samples.
        item = next(inputs, None)
        if item is None:
            break
        stats["claimed"] += 1
        stats["active_reads"] += 1
        stats["peak_reads"] = max(stats["peak_reads"], stats["active_reads"])
        try:
            raw = await asyncio.to_thread(read_file, item, args.read_delay)
        finally:
            stats["active_reads"] -= 1

        stats["active_process_calls"] += 1
        stats["peak_process_calls"] = max(
            stats["peak_process_calls"], stats["active_process_calls"])
        try:
            processed = await loop.run_in_executor(pool, transform, raw, args.rounds)
        finally:
            stats["active_process_calls"] -= 1
        raw = None  # Stop keeping a parent-side raw payload after transformation.
        if results.full():
            stats["full_queue_observations"] += 1
        await results.put(processed)  # A full queue pauses this coordinator.
        stats["peak_queue"] = max(stats["peak_queue"], results.qsize())
        processed = None
    await results.put(None)  # One completion marker per coordinator, on success.


async def consume(results, coordinator_count, batch_size, consumer_delay, stats):
    """A single consumer lives on the main event loop; retain only small features."""
    finished = 0
    batch = []
    seen = {}

    async def use_batch():
        # This wait models a busy downstream consumer; there is no GPU work here.
        await asyncio.sleep(consumer_delay)
        for job_id, data in batch:
            if job_id in seen:
                raise AssertionError(f"Duplicate sample {job_id}")
            seen[job_id] = crc32(data)  # A compact fingerprint sensitive to byte order.
        stats["batches"] += 1
        batch.clear()

    while finished < coordinator_count:
        item = await results.get()
        if item is None:
            finished += 1
        else:
            batch.append(item)
            item = None
            stats["peak_batch"] = max(stats["peak_batch"], len(batch))
            if len(batch) == batch_size:
                await use_batch()
    if batch:
        await use_batch()
    return seen


async def run_pipeline(items, pool, args, stats):
    results = asyncio.Queue(maxsize=args.capacity)
    inputs = iter(items)
    async with asyncio.TaskGroup() as group:
        consumer = group.create_task(consume(
            results, args.coordinators, args.batch_size, args.consumer_delay, stats))
        for _ in range(args.coordinators):
            group.create_task(coordinate(inputs, results, pool, args, stats))
    # TaskGroup waits for all tasks; a task failure cancels its siblings.
    return consumer.result()


def exception_leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in exception_leaves(child)]
    return [error]


def pipeline_demo(args):
    stats = {key: 0 for key in ("claimed", "active_reads", "peak_reads",
                              "active_process_calls", "peak_process_calls",
                              "peak_queue", "peak_batch", "batches",
                              "full_queue_observations")}
    with TemporaryDirectory(prefix="concurrency_lesson_") as temporary:
        items = make_inputs(Path(temporary), args.jobs, args.bytes)
        start = perf_counter()
        expected = sequential_baseline(items, args.rounds, args.read_delay,
                                       args.batch_size, args.consumer_delay)
        sequential_seconds = perf_counter() - start
        if args.fail_id is not None:
            items[args.fail_id][1].write_bytes(b"broken")

        start = perf_counter()
        pool = ProcessPoolExecutor(max_workers=args.workers,
                                   mp_context=mp.get_context("spawn"))
        errors = []
        try:
            try:
                actual = asyncio.run(run_pipeline(items, pool, args, stats))
            except ExceptionGroup as group:
                errors = exception_leaves(group)
                if (args.fail_id is None or
                        not all(isinstance(error, ValueError) for error in errors)):
                    raise
        finally:
            # Cancellation cannot stop an already running executor call. Our
            # finite calls finish; cancel queued calls, then join both processes.
            pool.shutdown(wait=True, cancel_futures=True)
        pipeline_seconds = perf_counter() - start

        assert stats["peak_reads"] <= args.coordinators
        assert stats["peak_process_calls"] <= args.coordinators
        assert stats["peak_queue"] <= args.capacity
        assert stats["peak_batch"] <= args.batch_size
        if args.fail_id is None:
            assert actual == expected
            status = "success"
        else:
            assert errors, "The requested corrupt-input experiment must fail."
            status = "expected_failure"

        return {"status": status, "jobs": args.jobs, "payload_bytes": args.bytes,
                "workers": args.workers, "coordinators": args.coordinators,
                "queue_capacity": args.capacity, "batch_size": args.batch_size,
                "sequential_seconds": sequential_seconds,
                "pipeline_seconds": pipeline_seconds,
                "all_results_match": status == "success",
                "errors": [str(error) for error in errors], "stats": stats,
                "executors_closed": True,
                "note": "Pipeline time includes process startup and shutdown."}


def positive_int(text):
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def nonnegative_float(text):
    number = float(text)
    if number < 0:
        raise argparse.ArgumentTypeError("Use zero or a positive number.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=positive_int, default=12)
    parser.add_argument("--bytes", type=positive_int, default=32_768)
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--workers", type=positive_int, default=2)
    parser.add_argument("--coordinators", type=positive_int, default=2)
    parser.add_argument("--capacity", type=positive_int, default=2)
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--read-delay", type=nonnegative_float, default=0.0)
    parser.add_argument("--consumer-delay", type=nonnegative_float, default=0.06)
    parser.add_argument("--fail-id", type=int, default=None)
    args = parser.parse_args()
    if args.fail_id is not None and not 0 <= args.fail_id < args.jobs:
        parser.error("--fail-id must identify one of the generated jobs.")
    print(json.dumps(pipeline_demo(args), indent=2))


if __name__ == "__main__":
    main()
