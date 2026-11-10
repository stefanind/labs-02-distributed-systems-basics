# Plain-language glossary

Use this page when a term is unfamiliar. The notebooks also explain terms where
they first appear. You do not need to memorize this list before starting.

[Return to the course guide](README.md).

## Running Python work

| Term | Plain meaning |
| --- | --- |
| Interpreter | The program that runs your Python code. |
| Kernel | The Python process that runs a notebook's cells and keeps their variables. Restarting it clears that state. |
| Virtual environment | A separate set of Python packages for a project. It helps keep one project's dependencies separate from another's. |
| Standard library | Modules supplied with Python, such as `threading`, `queue`, and `asyncio`. |
| CPU core | A part of the processor that can execute work. An operating system's logical CPU count can include hardware threads and need not equal physical cores. |
| Worker | A thread, process, or task assigned to do part of a job. Check which kind a particular example uses. |
| Process | A running program with its own ordinary Python objects and interpreter state. |
| Thread | A path of execution within a process. Threads in that process share its memory. |
| Concurrency | Several tasks are in progress during the same period, possibly taking turns. |
| Parallelism | Several tasks perform work at the same instant. |
| I/O | Input/output: reading files, receiving responses, or sending data outside your program. |
| Blocking call | A call that holds up its calling thread until it finishes or raises an error. |
| CPU-bound / I/O-bound | Mostly limited by calculation / mostly limited by waiting for input or output. |
| CPython / bytecode | The Python implementation most people install / the instructions it runs for your Python code. |
| GIL | Global Interpreter Lock. In a GIL-enabled CPython interpreter, one thread at a time executes Python bytecode. Some native operations release it. |
| Free-threaded build | A separate Python build that can run with the GIL disabled. Installing a recent Python version alone does not mean the GIL is disabled. |
| Spawn | Starting a fresh Python interpreter for a child process. The child needs definitions it can import. |
| Main guard | `if __name__ == "__main__":` runs code when a file is launched as a program, while avoiding that code during an ordinary import. |
| Context manager / `with` | A block that sets up a resource and cleans it up when the block ends, including when an error occurs. An executor's `with` block waits for its workers during cleanup. |

References: [Python threads](https://docs.python.org/3/library/threading.html),
[processes and spawn](https://docs.python.org/3/library/multiprocessing.html), and
[IPython notebook kernels](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

## Coordinating work and handling failures

| Term | Plain meaning |
| --- | --- |
| Pool / executor | A manager that assigns jobs to reusable workers. A worker limit is not automatically a limit on pending jobs. |
| Future | A handle for a job's eventual value or error. Calling `result()` waits if needed and exposes the outcome. |
| Coroutine | Work defined with `async def` that can pause when awaiting an unfinished operation. Calling the function creates a coroutine object. |
| Async task | A coroutine scheduled to run on an event loop. |
| Event loop | The coordinator that runs ready async tasks and resumes tasks whose waits finish. |
| `await` | Wait for an async outcome. If the operation is unfinished, the current task can pause so other tasks can run. |
| Lock / critical section | A lock lets one cooperating worker at a time enter the section that reads or changes protected data. |
| Invariant | A rule that must remain true, such as remaining stock plus sold stock equaling initial stock. |
| Race condition | A bug whose outcome depends on how operations from different workers overlap. |
| Deadlock | Workers wait on each other in a way that prevents progress. Holding locks in opposite orders is one cause. |
| Event | A flag for signaling readiness or stopping. |
| Barrier | A synchronization point. `threading.Barrier` waits for a set number of workers; `torch.cuda.synchronize()` waits for queued GPU work. |
| Semaphore | A limited set of permits. It can limit active operations, but does not itself limit how many waiting tasks you create. |
| Bounded queue | A queue with an item limit. A producer waits when it is full. Items already held by workers are outside that limit. |
| Backpressure | Slowing upstream production when downstream work cannot keep up. |
| Timeout | A limit on how long an operation waits. A future's timeout does not automatically stop its running worker. |
| Cooperative cancellation | A request to stop that running work must notice and handle at a safe point. |
| Structured concurrency / TaskGroup | Keeping related async tasks inside a scope that waits for them and handles failure together. `TaskGroup` cancels sibling tasks after an ordinary child failure and waits for cleanup. |
| ExceptionGroup | A container for several errors. `except*` handles matching errors inside it. |

References: [futures and executors](https://docs.python.org/3/library/concurrent.futures.html),
[thread coordination](https://docs.python.org/3/library/threading.html),
[async tasks and task groups](https://docs.python.org/3/library/asyncio-task.html), and
[async queues](https://docs.python.org/3/library/asyncio-queue.html).

## Moving data and using numerical libraries

| Term | Plain meaning |
| --- | --- |
| Serialization / pickle | Encoding an object as bytes so another process can reconstruct it. `pickle` is Python's usual tool for process-pool object transfer. |
| Shared memory | A memory block that several processes can access. It can reduce repeated copying, but access and cleanup still need clear rules. |
| Descriptor | Small information telling a worker where data lives and how to read it, rather than containing all the data itself. |
| Payload / metadata | The main data being processed / small facts describing it, such as an image's index or shape. |
| Chunk size | How many inputs a process pool groups into one scheduled unit. The worker function still handles individual inputs when using `map`. |
| Native code | Compiled code used inside a Python library, often written in C or C++. |
| Vectorization | Applying a library operation to a whole array instead of writing a Python loop for each value. It does not necessarily use several cores. |
| Matrix | A two-dimensional array of numbers. |
| dtype | The type used for an array's values. For example, `float32` uses four bytes per floating-point value; `int64` uses eight bytes per integer. |
| BLAS / backend | BLAS is a family of numerical routines, including matrix multiplication. A backend is the installed implementation used to run an operation. |
| Oversubscription | More runnable workers or threads compete than the available CPU capacity can use effectively. Nested Python and native thread pools can cause it. |
| Memory bandwidth | How much data can move to and from memory per second. More cores may not help if moving data is the limiting step. |
| CPU affinity | The set of CPUs the operating system allows a process to run on. |
| Fingerprint / checksum | A small value summarizing data. It can help detect changes, but different inputs can sometimes give the same value. |
| Floating-point tolerance | A small permitted numerical difference when comparing calculated results. Different calculation orders can produce slightly different rounding. |

References: [Python shared memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html),
[NumPy thread behavior](https://numpy.org/doc/stable/reference/thread_safety.html),
[native thread-pool controls](https://github.com/joblib/threadpoolctl), and
[NumPy comparison tolerances](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html).

## Measuring performance and feeding a model

| Term | Plain meaning |
| --- | --- |
| Baseline | A simple working version used for comparison. |
| Bottleneck | The slow part that limits the pace of the whole job. |
| Throughput | Completed items per second. |
| Latency / service time | Time from submitting an item until it finishes / time spent doing it after a worker starts. Latency also includes queue wait. |
| Median | The middle value after sorting measurements; for an even count, the average of the two middle values. |
| Profiling | Recording where time or allocations occur to locate expensive parts of a program. Profiling itself has a cost. |
| RAM / host memory | The computer's working memory. Host memory means memory on the CPU side, as distinct from GPU device memory. |
| KiB / MiB | 1,024 bytes / 1,048,576 bytes. These are different from decimal kilobytes and megabytes. |
| RSS | Resident set size: memory currently resident for a process. Parent-process RSS does not include the separate RSS of its child workers. |
| Tensor / shape | A container for numbers / the size of each of its dimensions. Shape `(2, 3, 16, 16)` can mean two images, three channels, 16 rows, and 16 columns. |
| Image channel | One value at each pixel for a component such as red, green, or blue. An RGB image has three channels. |
| Sample / batch / epoch | One training example / a group processed together / one pass through a dataset. |
| Dataset / DataLoader | Code that supplies samples / PyTorch's tool for fetching samples and grouping them into batches. |
| Sampler / sharding | Choosing the sample indices to request / dividing data into portions for different workers. |
| Prefetching | Preparing future batches before the consumer asks for them. With automatic DataLoader batching, `prefetch_factor` is batches per worker. |
| Persistent workers | DataLoader workers kept alive between epochs. Their dataset objects and random-number-generator state also stay alive. |
| RNG / seed | Random-number generator / a starting value for its sequence. Repeating a seed helps repeat an experiment when the other relevant settings stay the same. |
| Pinned memory | CPU memory kept resident for suitable device transfers. Pinning a tensor does not move it onto the GPU. |
| GPU / CUDA | A processor suited to many parallel numerical operations / NVIDIA's platform used for GPU work. CPU examples in this course do not require CUDA. |
| CUDA stream / synchronization | An ordered sequence of GPU operations / waiting until relevant device work finishes. Extra synchronization can reduce overlap between work stages. |
| Inference mode | A PyTorch mode for calculations without recording gradient tracking. It can reduce overhead when you do not need training gradients. |
| Gradient | Information about how a result changes when an input or model parameter changes. Training uses gradients to decide how to adjust parameters. |

References: [Python profiling](https://docs.python.org/3/library/profile.html),
[psutil process memory](https://psutil.readthedocs.io/stable/#psutil.Process.memory_info),
[PyTorch data loading](https://docs.pytorch.org/docs/stable/data.html),
[pinned memory and transfers](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html), and
[PyTorch inference mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html).
