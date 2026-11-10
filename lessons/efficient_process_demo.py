"""Finite multiprocessing experiments for notebook 10; no work runs on import."""

import argparse
import json
import multiprocessing as mp
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from statistics import median
from time import perf_counter


def cpu_score(task):
    """A deterministic Python loop. Every mode runs the same tasks."""
    job_id, iterations = task
    score = 0
    for value in range(iterations):
        score += ((value + job_id) ** 2) % 97
    return job_id, score


def timed_call(function):
    start = perf_counter()
    result = function()
    return result, perf_counter() - start


def batching_demo(jobs=12, iterations=120_000, workers=2, uneven=False):
    tasks = [(job_id, iterations * (4 if uneven and job_id >= jobs * 3 // 4 else 1))
             for job_id in range(jobs)]
    expected, sequential_seconds = timed_call(
        lambda: [cpu_score(task) for task in tasks])
    context = mp.get_context("spawn")

    # This full measurement includes constructing and closing a fresh pool.
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        fresh = list(pool.map(cpu_score, tasks, chunksize=1))
    fresh_seconds = perf_counter() - start
    assert fresh == expected

    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        # The first map starts the new workers. The timer excludes final shutdown.
        first, first_seconds = timed_call(
            lambda: list(pool.map(cpu_score, tasks, chunksize=1)))
        assert first == expected
        for chunksize in dict.fromkeys((1, 4, jobs)):
            measurements = []
            for _ in range(3):
                results, seconds = timed_call(
                    lambda: list(pool.map(cpu_score, tasks, chunksize=chunksize)))
                assert results == expected
                measurements.append(seconds)
            rows.append({"chunksize": chunksize,
                         "chunks": (jobs + chunksize - 1) // chunksize,
                         "median_seconds": median(measurements)})

    return {"demo": "batching", "jobs": jobs, "workers": workers,
            "uneven": uneven, "sequential_seconds": sequential_seconds,
            "fresh_pool_seconds": fresh_seconds,
            "first_map_seconds": first_seconds, "warm_pool": rows,
            "all_results_match": True}


def sum_array(array):
    import numpy as np
    return int(np.sum(array, dtype=np.int64))


def sum_shared_slice(descriptor):
    """Attach, read a slice, and close this worker's handle, even on failure."""
    import numpy as np
    name, shape, dtype, start, stop = descriptor
    handle = shared_memory.SharedMemory(name=name)
    view = None
    try:
        view = np.ndarray(shape, dtype=np.dtype(dtype), buffer=handle.buf)
        view.flags.writeable = False  # No worker in this lesson writes shared data.
        return int(np.sum(view[start:stop], dtype=np.int64))
    finally:
        del view  # Drop our array view before closing its backing handle.
        handle.close()


def shared_demo(mib=2, workers=2):
    import numpy as np
    count = mib * 1024 * 1024 // np.dtype(np.int64).itemsize
    array = np.arange(count, dtype=np.int64)
    edges = np.linspace(0, count, workers + 1, dtype=int)
    slices = [array[edges[i]:edges[i + 1]] for i in range(workers)]
    expected = [sum_array(part) for part in slices]
    context = mp.get_context("spawn")
    setup_start = perf_counter()
    owner = shared_memory.SharedMemory(create=True, size=array.nbytes)
    shared_view = None
    try:
        shared_view = np.ndarray(array.shape, dtype=array.dtype, buffer=owner.buf)
        shared_view[:] = array  # One initial copy into the shared block.
        shared_view.flags.writeable = False
        descriptors = [(owner.name, array.shape, array.dtype.str,
                        int(edges[i]), int(edges[i + 1])) for i in range(workers)]
        setup_seconds = perf_counter() - setup_start

        # Pickling here measures local encoding, not end-to-end process transfer.
        array_blob, array_pickle_seconds = timed_call(
            lambda: pickle.dumps(slices[0], protocol=pickle.HIGHEST_PROTOCOL))
        descriptor_blob, descriptor_pickle_seconds = timed_call(
            lambda: pickle.dumps(descriptors[0], protocol=pickle.HIGHEST_PROTOCOL))

        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            # A complete first pass pays process startup and NumPy imports.
            first, first_seconds = timed_call(
                lambda: list(pool.map(sum_array, slices)))
            assert first == expected
            timings = {"array_transfer": [], "shared_descriptor": []}
            # Alternate order between passes to reduce a fixed ordering bias.
            for pass_id in range(3):
                modes = [("array_transfer", sum_array, slices),
                         ("shared_descriptor", sum_shared_slice, descriptors)]
                if pass_id % 2:
                    modes.reverse()
                for label, function, arguments in modes:
                    results, seconds = timed_call(
                        lambda: list(pool.map(function, arguments)))
                    assert results == expected
                    timings[label].append(seconds)
        # The pool has finished using the block before its owner removes it.
        return {"demo": "shared", "workers": workers,
                "array_bytes": array.nbytes,
                "serialized_slice_bytes": len(array_blob),
                "serialized_descriptor_bytes": len(descriptor_blob),
                "local_array_pickle_seconds": array_pickle_seconds,
                "local_descriptor_pickle_seconds": descriptor_pickle_seconds,
                "shared_setup_seconds": setup_seconds,
                "first_array_map_seconds": first_seconds,
                "warm_array_seconds": median(timings["array_transfer"]),
                "warm_shared_seconds": median(timings["shared_descriptor"]),
                "all_results_match": True,
                "cleanup_policy": "workers close; owner closes and unlinks in finally"}
    finally:
        del shared_view
        try:
            owner.close()
        finally:
            # Try to remove the name even if closing the handle raises an error.
            # On Windows, closing all handles removes the block; unlink does nothing.
            owner.unlink()


def positive_int(text):
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=("batching", "shared"), default="batching")
    parser.add_argument("--jobs", type=positive_int, default=12)
    parser.add_argument("--iterations", type=positive_int, default=120_000)
    parser.add_argument("--workers", type=positive_int, default=2)
    parser.add_argument("--uneven", action="store_true")
    parser.add_argument("--mib", type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    if args.demo == "batching":
        report = batching_demo(args.jobs, args.iterations, args.workers, args.uneven)
    else:
        report = shared_demo(args.mib, args.workers)
    report["logical_cpus"] = os.cpu_count()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
