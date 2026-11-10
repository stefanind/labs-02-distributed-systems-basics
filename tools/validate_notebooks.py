"""Validate the course; optionally execute each notebook in a fresh kernel."""

import argparse
import ast
import asyncio
from pathlib import Path
import sys
from time import perf_counter

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_NAMES = [
    "01_concurrency_and_the_gil.ipynb",
    "02_threads_and_io.ipynb",
    "03_processes_and_cpu_work.ipynb",
    "04_asyncio.ipynb",
    "05_backpressure_and_bounded_queues.ipynb",
    "06_pytorch_dataloader.ipynb",
    "07_measuring_performance.ipynb",
    "08_safe_shared_state.ipynb",
    "09_failures_and_clean_shutdown.ipynb",
    "10_efficient_multiprocessing.ipynb",
    "11_native_library_parallelism.ipynb",
    "12_combining_concurrency_mechanisms.ipynb",
    "13_advanced_dataloader.ipynb",
]


class CourseNotebookClient(NotebookClient):
    """Run cells with the Python environment that launched this validator."""

    def create_kernel_manager(self):
        manager = super().create_kernel_manager()
        # Change this manager's in-memory specification, not an installed kernel.
        # A registered python3 kernel may otherwise point to another environment.
        manager.kernel_spec.argv = [
            sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
        ]
        manager.kernel_spec.env = {}
        return manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run all code cells.")
    parser.add_argument("--write-outputs", action="store_true", help="Save executed notebooks.")
    parser.add_argument("--start", type=int, default=1, help="First lesson number (default: 1).")
    parser.add_argument("--end", type=int, default=len(NOTEBOOK_NAMES), help="Last lesson number (default: 13).")
    parser.add_argument("--timeout", type=int, default=300, help="Seconds allowed per code cell (default: 300).")
    args = parser.parse_args()
    if args.write_outputs and not args.execute:
        parser.error("--write-outputs requires --execute")
    if not 1 <= args.start <= args.end <= len(NOTEBOOK_NAMES):
        parser.error("Use 1 <= --start <= --end <= 13")
    if args.timeout < 1:
        parser.error("--timeout must be a positive number of seconds")
    if args.execute and args.end >= 7 and sys.version_info < (3, 11):
        parser.error("Lessons 07-13 require Python 3.11 or newer")

    # Jupyter's Windows communication needs selector support. Set it in this
    # validator process, without changing any teaching notebook's event loop.
    if args.execute and sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    for name in NOTEBOOK_NAMES[args.start - 1:args.end]:
        path = ROOT / name
        start = perf_counter()
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{name}:cell-{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        if args.execute:
            print(f"Running {name} ...", flush=True)
            client = CourseNotebookClient(
                notebook,
                timeout=args.timeout,
                kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
                allow_errors=False,
                force_raise_errors=True,
                ipython_hist_file=":memory:",
            )
            client.execute()
            nbformat.validate(notebook)
            if args.write_outputs:
                nbformat.write(notebook, path)
        code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
        print(f"PASS {name}: {code_cells} code cells ({perf_counter() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
