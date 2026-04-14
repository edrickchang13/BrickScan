#!/usr/bin/env python3
"""
Benchmark CoreML model inference performance on macOS.

This script loads a CoreML model (.mlpackage or .mlmodel) and runs
inference benchmarks to measure latency, throughput, and percentile
performance on macOS using coremltools predict().

Note: This benchmarks CPU/GPU performance on macOS. On-device performance
on iPhone will be different due to Apple Neural Engine acceleration and
device-specific constraints.

Features:
- Warm-up runs to stabilize performance
- 100 inference passes with random 224x224 RGB images
- Latency statistics: mean, median (p50), p95, p99
- Throughput calculation (inferences/sec)
- Model size reporting
- Detailed timing breakdown

Usage:
    python benchmark_coreml.py --model-path lego_classifier.mlpackage

Example:
    python benchmark_coreml.py \\
        --model-path models/lego_classifier.mlpackage \\
        --warmup 10 \\
        --num-runs 100
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Try to import required libraries
try:
    import coremltools as ct
except ImportError:
    print(
        "Error: coremltools is required.\n"
        "Install it with: pip install coremltools\n"
        "For full support: pip install 'coremltools>=7.0'"
    )
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CoreMLBenchmark:
    """Benchmark CoreML model inference performance."""

    INPUT_SIZE = 224
    INPUT_CHANNELS = 3

    def __init__(
        self,
        model_path: str,
        warmup_runs: int = 10,
        num_runs: int = 100,
    ):
        """
        Initialize benchmark.

        Args:
            model_path: Path to CoreML model (.mlpackage or .mlmodel)
            warmup_runs: Number of warm-up runs before actual benchmark
            num_runs: Number of inference runs for benchmarking

        Raises:
            FileNotFoundError: If model doesn't exist
        """
        self.model_path = Path(model_path)
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = None
        self.model_size = None
        self.input_name = None
        self.output_name = None
        self.latencies: List[float] = []

        logger.info(f"Initialized benchmark for {self.model_path.name}")
        logger.info(f"Warm-up runs: {warmup_runs}, Benchmark runs: {num_runs}")

    def load_model(self) -> None:
        """Load CoreML model."""
        logger.info(f"Loading CoreML model from {self.model_path}")

        try:
            self.model = ct.models.MLModel(str(self.model_path))
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        logger.info("CoreML model loaded successfully")

        # Get model size
        if self.model_path.suffix == ".mlmodel":
            self.model_size = self.model_path.stat().st_size
        else:
            # For .mlpackage (directory), sum all files
            total_size = 0
            for file in self.model_path.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
            self.model_size = total_size

        logger.info(f"Model size: {self.model_size / (1024*1024):.2f} MB")

    def get_input_output_names(self) -> Tuple[str, str]:
        """Extract input and output names from model spec."""
        spec = self.model.get_spec()

        # Get input name
        input_desc = spec.description.input
        if len(input_desc) == 0:
            raise ValueError("Model has no inputs")
        self.input_name = input_desc[0].name

        # Get output name
        output_desc = spec.description.output
        if len(output_desc) == 0:
            raise ValueError("Model has no outputs")
        self.output_name = output_desc[0].name

        logger.info(f"Input: {self.input_name}")
        logger.info(f"Output: {self.output_name}")

        return self.input_name, self.output_name

    def create_random_input(self) -> np.ndarray:
        """Create random input image in HWC format (CoreML expects)."""
        # CoreML expects HWC format for image inputs
        return np.random.rand(self.INPUT_SIZE, self.INPUT_SIZE, self.INPUT_CHANNELS).astype(
            np.float32
        )

    def warmup(self) -> None:
        """Run warm-up inferences to stabilize performance."""
        logger.info(f"Running {self.warmup_runs} warm-up inferences")

        for i in range(self.warmup_runs):
            dummy_input = self.create_random_input()
            input_dict = {self.input_name: dummy_input}

            try:
                _ = self.model.predict(input_dict)
            except Exception as e:
                logger.error(f"Warm-up inference {i+1} failed: {e}")
                raise

        logger.info("Warm-up complete")

    def benchmark(self) -> None:
        """Run benchmark inferences and measure latency."""
        logger.info(f"Starting benchmark with {self.num_runs} inference runs")

        self.latencies = []

        for i in range(self.num_runs):
            dummy_input = self.create_random_input()
            input_dict = {self.input_name: dummy_input}

            try:
                start_time = time.time()
                output_dict = self.model.predict(input_dict)
                end_time = time.time()

                latency_ms = (end_time - start_time) * 1000
                self.latencies.append(latency_ms)

                if (i + 1) % 20 == 0:
                    logger.info(f"Completed {i+1}/{self.num_runs} runs")

            except Exception as e:
                logger.error(f"Benchmark inference {i+1} failed: {e}")
                raise

        logger.info("Benchmark complete")

    def compute_statistics(self) -> dict:
        """Compute latency and throughput statistics."""
        if not self.latencies:
            raise ValueError("No latency data available")

        latencies = sorted(self.latencies)

        # Latency statistics (in milliseconds)
        mean_latency = np.mean(self.latencies)
        median_latency = np.median(self.latencies)
        p95_latency = np.percentile(self.latencies, 95)
        p99_latency = np.percentile(self.latencies, 99)
        min_latency = np.min(self.latencies)
        max_latency = np.max(self.latencies)
        std_latency = np.std(self.latencies)

        # Throughput (inferences per second)
        mean_latency_sec = mean_latency / 1000
        throughput = 1.0 / mean_latency_sec

        # Total time
        total_time_sec = sum(self.latencies) / 1000

        stats = {
            "mean_latency_ms": mean_latency,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "std_latency_ms": std_latency,
            "throughput_inferences_per_sec": throughput,
            "total_time_sec": total_time_sec,
        }

        return stats

    def print_report(self, stats: dict) -> None:
        """Print detailed benchmark report."""
        logger.info("=" * 80)
        logger.info("COREML BENCHMARK REPORT")
        logger.info("=" * 80)

        print(f"Model:                  {self.model_path.name}")
        print(f"Model Size:             {self.model_size / (1024*1024):.2f} MB")
        print(f"Input Size:             224 x 224 x 3 (RGB)")
        print()

        print("LATENCY STATISTICS (milliseconds)")
        print("-" * 80)
        print(f"  Mean:                 {stats['mean_latency_ms']:.3f} ms")
        print(f"  Median (p50):         {stats['median_latency_ms']:.3f} ms")
        print(f"  p95:                  {stats['p95_latency_ms']:.3f} ms")
        print(f"  p99:                  {stats['p99_latency_ms']:.3f} ms")
        print(f"  Min:                  {stats['min_latency_ms']:.3f} ms")
        print(f"  Max:                  {stats['max_latency_ms']:.3f} ms")
        print(f"  Std Dev:              {stats['std_latency_ms']:.3f} ms")
        print()

        print("THROUGHPUT")
        print("-" * 80)
        print(f"  Inferences/sec:       {stats['throughput_inferences_per_sec']:.2f}")
        print()

        print("EXECUTION SUMMARY")
        print("-" * 80)
        print(f"  Total Runs:           {self.num_runs}")
        print(f"  Warm-up Runs:         {self.warmup_runs}")
        print(f"  Total Time:           {stats['total_time_sec']:.2f} seconds")
        print()

        print("NOTES")
        print("-" * 80)
        print("  - macOS CPU/GPU performance differs from iPhone Neural Engine")
        print("  - On-device performance will be better due to ANE acceleration")
        print("  - Input: 224x224 RGB images (random values)")
        print("  - Input format: HWC (Height, Width, Channel)")
        print()

        logger.info("=" * 80)

    def run(self) -> bool:
        """Run complete benchmark pipeline."""
        try:
            self.load_model()
            self.get_input_output_names()
            self.warmup()
            self.benchmark()

            stats = self.compute_statistics()
            self.print_report(stats)

            logger.info("Benchmark completed successfully")
            return True

        except Exception as e:
            logger.error(f"Benchmark failed: {e}", exc_info=True)
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark CoreML model inference performance on macOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python benchmark_coreml.py --model-path lego_classifier.mlpackage

  # Custom number of runs
  python benchmark_coreml.py \\
    --model-path lego_classifier.mlpackage \\
    --num-runs 200 \\
    --warmup 20

  # For .mlmodel format
  python benchmark_coreml.py --model-path lego_classifier.mlmodel
        """,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to CoreML model (.mlpackage or .mlmodel)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of inference runs for benchmarking (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warm-up runs before benchmarking (default: 10)",
    )

    args = parser.parse_args()

    try:
        benchmark = CoreMLBenchmark(
            model_path=args.model_path,
            warmup_runs=args.warmup,
            num_runs=args.num_runs,
        )

        success = benchmark.run()
        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
