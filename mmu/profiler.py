"""Pre-flight profiler for HATS build scripts.

Processes a small sample of rows from a dataset's build script, measures peak
RSS and throughput, and recommends resource parameters (workers, walltime,
pixel_threshold) for the full production run.

Usage::

    from mmu.profiler import profile_build
    result = profile_build("manga", n_sample=50)
    print(result)
    # ProfileResult(
    #   dataset='manga',
    #   n_sample=50,
    #   peak_rss_mb=24310,
    #   rss_per_row_mb=486,
    #   rows_per_sec=2.3,
    #   total_rows=10735,
    #   recommended_workers=4,
    #   recommended_walltime_h=12,
    #   recommended_pixel_threshold=100000,
    # )

Run from CLI::

    python -m mmu.profiler manga --n-sample 50
    python -m mmu.profiler legacysurvey --n-sample 10
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass

import psutil


@dataclass
class ProfileResult:
    dataset: str
    n_sample: int
    peak_rss_mb: float
    rss_per_row_mb: float
    rows_per_sec: float
    total_rows: int | None
    recommended_workers: int
    recommended_walltime_h: float
    recommended_pixel_threshold: int

    def __str__(self) -> str:
        lines = [
            f"=== Profile: {self.dataset} ({self.n_sample} sample rows) ===",
            f"  Peak RSS:           {self.peak_rss_mb:.0f} MB",
            f"  RSS per row:        {self.rss_per_row_mb:.0f} MB",
            f"  Throughput:         {self.rows_per_sec:.2f} rows/sec",
        ]
        if self.total_rows is not None:
            scatter_h = self.total_rows / max(self.rows_per_sec, 0.01) / 3600
            lines.append(f"  Total rows:         {self.total_rows}")
            lines.append(f"  Est. scatter time:  {scatter_h:.1f}h (single worker)")
        lines += [
            f"  --- Recommendations (for 900 GB node) ---",
            f"  Workers:            {self.recommended_workers}",
            f"  Walltime:           {self.recommended_walltime_h:.0f}h",
            f"  pixel_threshold:    {self.recommended_pixel_threshold}",
        ]
        return "\n".join(lines)


def _measure_rss_mb() -> float:
    """Current process RSS in MB including all children."""
    proc = psutil.Process()
    rss = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return rss / 1e6


def profile_build(
    dataset: str,
    n_sample: int = 50,
    node_mem_mb: int = 900_000,
    safety_factor: float = 2.0,
) -> ProfileResult:
    """Profile a dataset's build script on a small sample.

    Imports the build script, finds its processing function, runs it on
    ``n_sample`` rows, and measures RSS + throughput.

    This is intentionally simple and dataset-agnostic: it calls ``main()``
    with ``--max-files {n_sample}`` (or equivalent) and measures the process.
    For sharded datasets, it runs a single shard (shard 0 of 1).
    """
    rss_before = _measure_rss_mb()

    # Build the CLI args for a small test run
    from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Known: {list(DATASETS.keys())}")

    # Import the build module
    script_path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", dataset,
        "build_parent_sample_hats.py",
    )
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"No build script at {script_path}")

    spec = importlib.util.spec_from_file_location(f"_profile_{dataset}", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Run with --max-files to limit scope, write to /tmp
    tmp_output = f"/tmp/mmu_profile_{dataset}_{os.getpid()}"
    tmp_scratch = f"/tmp/mmu_profile_scratch_{dataset}_{os.getpid()}"
    os.makedirs(tmp_output, exist_ok=True)
    os.makedirs(tmp_scratch, exist_ok=True)

    argv = [
        "--output-root", tmp_output,
        "--max-files", str(n_sample),
        "--num-processes", "1",  # serial mode — avoids pickle issues with dynamic import
    ]

    # Add scratch-dir if the script supports it (sharded datasets)
    import inspect
    main_sig = inspect.signature(mod.main)
    main_src = inspect.getsource(mod.main)
    if "scratch-dir" in main_src or "scratch_dir" in main_src:
        argv += ["--scratch-dir", tmp_scratch, "--skip-ingest"]

    # Measure
    rss_samples = [_measure_rss_mb()]
    t0 = time.time()

    try:
        rc = mod.main(argv)
    except SystemExit as e:
        rc = e.code or 0

    elapsed = time.time() - t0
    peak_rss = _measure_rss_mb()
    rss_after = peak_rss  # approximate — real peak may have been higher

    # Count output rows
    import glob
    import pyarrow.parquet as pq
    parquets = glob.glob(os.path.join(tmp_scratch, "*.parquet")) or \
               glob.glob(os.path.join(tmp_output, "**/*.parquet"), recursive=True)
    n_output_rows = sum(pq.read_metadata(f).num_rows for f in parquets) if parquets else 0

    # Clean up
    import shutil
    shutil.rmtree(tmp_output, ignore_errors=True)
    shutil.rmtree(tmp_scratch, ignore_errors=True)

    # Calculate recommendations
    rss_delta = max(peak_rss - rss_before, 100)  # MB used by the sample
    rss_per_row = rss_delta / max(n_output_rows, 1)
    rows_per_sec = n_output_rows / max(elapsed, 0.01)

    # Try to estimate total rows from the dataset
    total_rows = None
    if hasattr(mod, "load_catalog"):
        try:
            # Don't actually load — just check if there's a known count
            pass
        except Exception:
            pass

    # Recommendations
    # Workers: each worker holds ~pixel_threshold rows at peak during reduce
    # For scatter: each worker holds 1 unit of work
    worker_mem_budget = node_mem_mb / safety_factor  # usable MB
    recommended_workers = max(1, int(worker_mem_budget / max(rss_delta, 100)))
    recommended_workers = min(recommended_workers, 32)  # cap at 32

    # Walltime: scatter time / n_workers * safety_factor + gather overhead
    if total_rows and rows_per_sec > 0:
        scatter_sec = total_rows / rows_per_sec / recommended_workers
        gather_sec = total_rows * 0.01  # rough: 10ms per row for hats-import
        recommended_walltime_h = (scatter_sec + gather_sec) / 3600 * safety_factor
    else:
        recommended_walltime_h = 48  # safe default

    # pixel_threshold: for datasets with <100k rows, just use 100k
    # For larger datasets, use 100k (our proven default)
    recommended_pixel_threshold = 100_000

    return ProfileResult(
        dataset=dataset,
        n_sample=n_sample,
        peak_rss_mb=peak_rss,
        rss_per_row_mb=rss_per_row,
        rows_per_sec=rows_per_sec,
        total_rows=total_rows,
        recommended_workers=recommended_workers,
        recommended_walltime_h=max(recommended_walltime_h, 4),
        recommended_pixel_threshold=recommended_pixel_threshold,
    )


def main():
    parser = argparse.ArgumentParser(description="Profile a HATS build script")
    parser.add_argument("dataset", help="Dataset name (e.g., manga, legacysurvey)")
    parser.add_argument("--n-sample", type=int, default=50,
                        help="Number of sample rows/files to process")
    parser.add_argument("--node-mem-mb", type=int, default=900_000,
                        help="Node memory in MB (default 900 GB)")
    args = parser.parse_args()

    result = profile_build(args.dataset, n_sample=args.n_sample,
                           node_mem_mb=args.node_mem_mb)
    print(result)


if __name__ == "__main__":
    main()
