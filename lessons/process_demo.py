"""Small, standalone process examples used by notebook 03.

Run from the project folder:
    python lessons/process_demo.py --demo memory
    python lessons/process_demo.py --demo compare --workers 2

All worker functions live at the top level so spawned Python interpreters
can find them. Importing this module does not start any processes.
"""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from time import perf_counter


def append_in_child(names):
    """Change this process's copy of a regular Python list."""
    names.append("child")
    print(f"Child PID {os.getpid()}: {names}", flush=True)


def memory_demo():
    names = ["parent"]
    context = mp.get_context("spawn")
    process = context.Process(target=append_in_child, args=(names,))
    print(f"Parent PID {os.getpid()}, before: {names}", flush=True)
    process.start()
    process.join(timeout=20)  # Wait for the child to finish.
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("The small child-process demo took too long.")
    if process.exitcode != 0:
        raise RuntimeError(f"Child exited with code {process.exitcode}")
    print(f"Parent PID {os.getpid()}, after:  {names}")
    assert names == ["parent"]


def cpu_score(task):
    """Synthetic CPU work: repeated Python arithmetic, not image decoding."""
    job_id, iterations = task
    score = 0
    for value in range(iterations):
        score += ((value + job_id) ** 2) % 97
    return job_id, score


def compare_demo(iterations=120_000, jobs=8, workers=2, chunksize=1):
    tasks = [(job_id, iterations) for job_id in range(jobs)]

    start = perf_counter()
    sequential = [cpu_score(task) for task in tasks]
    sequential_seconds = perf_counter() - start

    start = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        threaded = list(pool.map(cpu_score, tasks))
    thread_seconds = perf_counter() - start

    context = mp.get_context("spawn")
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        processed = list(pool.map(cpu_score, tasks, chunksize=chunksize))
    process_seconds = perf_counter() - start

    assert sequential == threaded == processed
    print(f"Jobs: {jobs}; iterations per job: {iterations:,}; workers: {workers}")
    print(f"Process chunksize: {chunksize}; logical CPUs reported: {os.cpu_count()}")
    print(f"Sequential: {sequential_seconds:.4f} s")
    print(f"Threads:    {thread_seconds:.4f} s")
    print(f"Processes:  {process_seconds:.4f} s")
    print("All three methods returned the same results.")
    print(f"First two results: {processed[:2]}")
    print("Pool timings include worker startup and shutdown.")


def cpu_score_batch(batch):
    """An application batch is explicitly a list of individual tasks."""
    return [cpu_score(task) for task in batch]


def batch_demo(iterations=120_000, jobs=8, workers=2, batch_size=2):
    tasks = [(job_id, iterations) for job_id in range(jobs)]
    batches = [tasks[index:index + batch_size]
               for index in range(0, len(tasks), batch_size)]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        nested_results = list(pool.map(cpu_score_batch, batches, chunksize=1))
    # Each worker call returns a list; combine those lists into one list.
    results = [item for batch in nested_results for item in batch]
    assert results == [cpu_score(task) for task in tasks]
    print(f"{len(tasks)} jobs grouped into {len(batches)} application batches.")
    print(f"Items per returned batch: {[len(batch) for batch in nested_results]}")
    print("Batched results match the sequential results.")


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=["memory", "compare", "batches"],
                        default="compare")
    parser.add_argument("--iterations", type=positive_int, default=120_000)
    parser.add_argument("--jobs", type=positive_int, default=8)
    parser.add_argument("--workers", type=positive_int, default=2)
    parser.add_argument("--chunksize", type=positive_int, default=1)
    parser.add_argument("--batch-size", type=positive_int, default=2)
    args = parser.parse_args()
    if args.demo == "memory":
        memory_demo()
    elif args.demo == "compare":
        compare_demo(args.iterations, args.jobs, args.workers, args.chunksize)
    else:
        batch_demo(args.iterations, args.jobs, args.workers, args.batch_size)


if __name__ == "__main__":
    main()
