import sys


BENCHMARK_ROOT = _resolve("benchmark_root", f"{HATS_ROOT}/benchmarks").rstrip("/")
validate_hats_root(BENCHMARK_ROOT, benchmark=True)
BENCHMARKS = ("gz10", "desi_provabgs", "galaxy_cruise", "euclid_q1_morphology")


rule benchmark_datasets:
    input:
        expand(f"{BENCHMARK_ROOT}/{{benchmark}}", benchmark=BENCHMARKS),


rule build_benchmark_dataset:
    input:
        script="scripts/benchmark_datasets/build.py",
    output:
        directory(f"{BENCHMARK_ROOT}/{{benchmark}}"),
    wildcard_constraints:
        benchmark="|".join(BENCHMARKS),
    params:
        python=sys.executable,
        root=BENCHMARK_ROOT,
    threads: 4
    resources:
        mem_mb=8_000,
        runtime=60,
    shell:
        "{params.python:q} {input.script:q} {wildcards.benchmark:q} {params.root:q} --workers {threads}"
