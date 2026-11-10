# Learn Python concurrency

Thirteen notebooks for learning Python concurrency, with an emphasis on machine learning data pipelines. Lessons 01–06 cover the basics; lessons 07–13 continue into measurement, coordination, reliable shutdown, efficient processes, native library threads, bounded pipelines, and PyTorch tuning. Each lesson explains new terms, asks you to predict an outcome, and gives you small experiments and exercises with worked answers.

The [plain-language glossary](GLOSSARY.md) is a reference for unfamiliar terms. Read it as needed; the lessons also define terms where they use them.

Start with [01 · Concurrency and the GIL](01_concurrency_and_the_gil.ipynb). Work in order the first time. Allow roughly three hours for the first six lessons and four to six hours for the continuation, including experiments. Take one lesson at a time. The continuation ends at lesson 13; a capstone is not included.

| Lesson | What you will learn | Approximate study time |
| --- | --- | --- |
| [01 · Concurrency and the GIL](01_concurrency_and_the_gil.ipynb) | Waiting versus calculating, concurrency versus parallelism, the GIL, and measuring a baseline | 20 min |
| [02 · Threads and I/O](02_threads_and_io.ipynb) | `threading`, thread pools, results, errors, and shared data | 25–30 min |
| [03 · Processes and CPU work](03_processes_and_cpu_work.ipynb) | Processes, process pools, separate memory, startup, and data transfer | 25–35 min |
| [04 · Asyncio](04_asyncio.ipynb) | Coroutines, `await`, tasks, blocking calls, and concurrency limits | 25–30 min |
| [05 · Backpressure](05_backpressure_and_bounded_queues.ipynb) | Producers, consumers, bounded queues, and batch memory | 25 min |
| [06 · PyTorch DataLoader](06_pytorch_dataloader.ipynb) | Worker assignment, iterable datasets, pinned memory, and prefetching | 30–40 min |
| [07 · Measuring performance](07_measuring_performance.ipynb) | Throughput, latency, queue wait, profiling, and memory measurements | 35–45 min |
| [08 · Safe shared state](08_safe_shared_state.ipynb) | Multi-step updates, locks, ownership, signals, and deadlocks | 30–45 min |
| [09 · Failures and clean shutdown](09_failures_and_clean_shutdown.ipynb) | Exceptions, timeouts, cooperative cancellation, and task groups | 30–45 min |
| [10 · Efficient multiprocessing](10_efficient_multiprocessing.ipynb) | Reusable pools, chunks, load balance, serialization, and shared arrays | 35–45 min |
| [11 · Native library parallelism](11_native_library_parallelism.ipynb) | Vectorization, NumPy/PyTorch threads, and competing thread pools | 35–45 min |
| [12 · Combining mechanisms](12_combining_concurrency_mechanisms.ipynb) | File-reading threads, preprocessing processes, bounded queues, and failure cleanup | 35–45 min |
| [13 · Advanced DataLoader](13_advanced_dataloader.ipynb) | Fair tuning, persistent workers, reproducible randomness, and stage timing | 35–45 min |

## Setup

Use Python 3.11 or newer for the full series. The original lessons 01–06 also support Python 3.10. You need basic functions, loops, lists, and dictionaries; later lessons explain context managers, cleanup, and arrays as they use them. The first five lessons use Python's standard library for their examples. Jupyter supplies the notebook interface. You can read every lesson without running it.

A **virtual environment** gives this course its own set of installed packages. `pip` installs those packages. A notebook's **kernel** is the Python process that runs its cells. Use the course environment as the kernel so it can find the packages you install.

From this folder on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name concurrency-course --display-name "Python (Concurrency course)"
.\.venv\Scripts\python.exe -m jupyterlab .
```

On macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --sys-prefix --name concurrency-course --display-name "Python (Concurrency course)"
.venv/bin/python -m jupyterlab .
```

For the continuation (07–13), install its small additional dependencies in the same environment. On Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-advanced.txt
```

This adds NumPy for arrays, psutil for process-memory measurements, and threadpoolctl for inspecting native computation threads. Lessons 08–09 use only the standard library beyond Jupyter.

For the PyTorch experiments in lessons 06, 11, and 13, install PyTorch in the same environment. Those sections explain their skips if it is missing. On Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-torch.txt
```

On macOS/Linux use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`. The lesson runs on a CPU. If you want a CUDA installation, use the command for your system from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). The lesson checks for CUDA before using it.

In Jupyter's kernel menu, choose **Python (Concurrency course)**. The installation command above adds this name inside the course environment. You can also open the notebooks in VS Code with its Jupyter extension and select the `.venv` environment as the kernel. [IPython's environment and kernel guide](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

## How to study

1. Read the explanation and predict what the next experiment will show.
2. Run cells from top to bottom. Each notebook starts fresh; it does not need variables from another notebook.
3. Change one small setting, such as worker count or queue capacity, and run again.
4. Try the exercises before opening the worked answers.
5. Use **Restart Kernel and Run All Cells** to check your understanding and reset state.

The examples use tiny synthetic data and local temporary files. They do not download datasets, contact APIs, or require a GPU. A sleep stands in for waiting or a pipeline stage only when the lesson says so. Real CPU examples perform arithmetic. Timing values are observations from one run, not promises of a speedup.

## Why there are helper scripts

Keep the `lessons/` folder beside the notebooks. Process and DataLoader experiments run guarded Python files there in a separate interpreter. The notebooks explain and show the relevant definitions before using them. Lesson 11 also runs a small helper so each PyTorch CPU thread configuration starts fresh.

New processes need importable definitions and a guarded entry point (`if __name__ == "__main__":`). Notebook cells are awkward places to provide those, especially on Windows. The scripts explicitly use the `spawn` start method so the examples use the same approach across operating systems. The notebook launcher uses the kernel's Python interpreter. [Python multiprocessing guidance](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods).

Asyncio cells use Jupyter's top-level `await`. Do not wrap them in `asyncio.run()` inside Jupyter; its event loop is already running. When adapting a lesson to a regular `.py` script, put async work in `async def main()` and call `asyncio.run(main())` at the entry point. [IPython async support](https://ipython.readthedocs.io/en/stable/interactive/autoawait.html).

## Check the notebooks

The validator checks notebook structure. With `--execute`, it also runs every lesson in a fresh kernel, including its result checks. From the environment that has the requirements installed:

```powershell
.\.venv\Scripts\python.exe tools/validate_notebooks.py
.\.venv\Scripts\python.exe tools/validate_notebooks.py --execute
.\.venv\Scripts\python.exe tools/validate_notebooks.py --start 7 --end 13 --execute
```

On macOS/Linux use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

The validator checks notebook structure and compiles code cells, including top-level `await`. Execution uses the Python environment that launched the validator, even if an installed `python3` kernel points somewhere else. It does not change your installed kernel settings. Add `--write-outputs` with `--execute` to save new outputs. Running without it leaves the notebooks unchanged. `--start` and `--end` include both endpoints. Each code cell has a 300-second limit; use `--timeout 600` if startup is unusually slow. The full run takes longer than individual cells because process workers and PyTorch have startup costs. An environment that forbids subprocesses or local kernel communication cannot run these examples fully.

For worker errors, first run from this project folder, check that the helper scripts are present, and verify that Jupyter is using the intended Python environment. In lesson 06, begin with `num_workers=0` to see dataset errors directly. A slow process-pool result on tiny inputs is expected: starting workers can cost more than the work they do.

The main references are linked beside the explanations in each notebook: official Python, IPython, and PyTorch documentation.

All 13 lessons were reviewed and executed on Windows with Python 3.13.7, NumPy 2.2.6, and PyTorch 2.9.1 on CPU. Their outputs are saved for inspection. Missing-PyTorch skips and the CUDA-unavailable path were checked; actual GPU execution was unavailable. Recorded timings describe that run and will change on another machine.

The full run printed one native ZeroMQ connection-reset assertion on Windows. All 119 code cells completed without notebook error outputs, and a focused rerun of lessons 09–10 passed without the assertion. The evidence suggests a Jupyter kernel shutdown problem; [similar Windows failures are reported by libzmq](https://github.com/zeromq/libzmq/issues/1808). Its exact cause was not confirmed.
