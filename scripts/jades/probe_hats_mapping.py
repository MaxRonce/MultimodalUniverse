"""Map ONE existing Parquet file before submitting another full JADES import.

Run on a compute node, with --probe-root on shared Ceph. Without a scheduler
file this starts one local Dask worker in a separate process. An external
scheduler can also be tested without changing any extraction or HATS output.
"""

import argparse
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import socket
import sys
import tempfile

import cloudpickle
import pyarrow.parquet as pq
from dask.distributed import Client
from hats_import.catalog.file_readers import ParquetPyarrowReader
from hats_import.catalog.map_reduce import map_to_pixels

from mmu.hats_import import _MappingClient, _sparse_map_to_pixels


def runtime_info():
    return {"host": socket.gethostname(), "python": sys.version.split()[0],
            "versions": {p: version(p) for p in ("hats", "hats-import", "dask", "distributed", "pyarrow")},
            "mapper_module": inspect.getfile(_sparse_map_to_pixels),
            "mapper_sha256": hashlib.sha256(inspect.getsource(_sparse_map_to_pixels).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--scheduler-file", type=Path)
    parser.add_argument("--highest-order", type=int, default=12)
    args = parser.parse_args()
    if not 0 <= args.highest_order <= 29:
        parser.error("--highest-order must be between 0 and 29")
    source = next(args.scratch_dir.rglob("*.parquet"))
    probe = Path(tempfile.mkdtemp(prefix="hats_mapping_probe_", dir=args.probe_root))
    reader = probe / "reader.pickle"
    reader.write_bytes(cloudpickle.dumps(ParquetPyarrowReader(chunksize=4, pre_buffer=False)))
    with pq.ParquetFile(source) as parquet:
        expected = parquet.metadata.num_rows
    print(f"Input: {source}; expected rows: {expected}; probe: {probe}", flush=True)
    controller = runtime_info()
    print("Client: " + json.dumps(controller, sort_keys=True), flush=True)
    options = ({"scheduler_file": str(args.scheduler_file)} if args.scheduler_file else
               {"processes": True, "n_workers": 1, "threads_per_worker": 1,
                "memory_limit": "4GiB", "dashboard_address": None})
    with Client(**options) as client:
        workers = client.run(runtime_info)
        print("Workers: " + json.dumps(workers, sort_keys=True), flush=True)
        if not workers or any(info["versions"] != controller["versions"]
                              or info["mapper_sha256"] != controller["mapper_sha256"]
                              for info in workers.values()):
            raise RuntimeError("Client/worker package or mapper mismatch")
        checked = _MappingClient(client)
        future = checked.submit(map_to_pixels, input_file=str(source),
                                pickled_reader_file=str(reader), resume_path=str(probe),
                                mapping_key="map_0", highest_order=args.highest_order,
                                ra_column="ra", dec_column="dec")
        result = future.result(timeout=600)
        checked.finish_mapping()
        if result["rows"] != expected:
            raise RuntimeError(f"Mapping lost rows: {result}")
        print("PASS: ONE mapping task executed, checkpoint read from worker AND client; "
              + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
