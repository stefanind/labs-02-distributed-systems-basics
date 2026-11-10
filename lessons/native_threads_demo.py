"""A fresh interpreter makes each optional PyTorch CPU setting independent."""

import argparse
import json
import statistics
from time import perf_counter


def measure_torch(threads):
    import torch

    # Configure CPU operation threads before allocating or computing tensors.
    torch.set_num_threads(threads)
    # A fixed seed generates the same input numbers for every configuration.
    generator = torch.Generator().manual_seed(7)
    left = torch.randn((320, 320), generator=generator, dtype=torch.float64)
    right = torch.randn((320, 320), generator=generator, dtype=torch.float64)
    timings = []
    # This CPU example needs no bookkeeping for training gradients.
    with torch.inference_mode():
        torch.mm(left, right)  # Untimed warm-up in the same mode as the measured work.
        for _ in range(3):
            start = perf_counter()
            for _ in range(8):
                result = torch.mm(left, right)
            timings.append(perf_counter() - start)
    return {
        "threads": torch.get_num_threads(),
        "seconds": statistics.median(timings),
        # A compact comparison of the last result, not a full correctness proof.
        "checksum": result.sum().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    print(json.dumps(measure_torch(args.threads)))


if __name__ == "__main__":
    main()
