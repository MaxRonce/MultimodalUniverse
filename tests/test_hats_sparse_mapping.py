"""Exercise real hats-import readers and checkpoint format, including retries."""

import cloudpickle
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dask.distributed import Client
from hats.pixel_math import healpix_shim as hp
from hats.pixel_math.sparse_histogram import SparseHistogram
from hats_import.catalog.file_readers import ParquetPyarrowReader
from hats_import.catalog.map_reduce import map_to_pixels

from mmu.hats_import import _MappingClient, _sparse_map_to_pixels, write_hats_from_parquet_dir


def inputs(tmp_path):
    path = tmp_path / "input.parquet"
    ra = np.array([53.0, 53.00001, 53.2, 189.0, 189.00001])
    dec = np.array([-27.0, -27.0, -27.0, 62.0, 62.0])
    pq.write_table(pa.table({"ra": ra, "dec": dec, "image": [b"cutout"] * 5}),
                   path, row_group_size=2)
    reader = tmp_path / "reader.pickle"
    reader.write_bytes(cloudpickle.dumps(ParquetPyarrowReader(chunksize=2)))
    return dict(input_file=str(path), pickled_reader_file=str(reader),
                resume_path=str(tmp_path), mapping_key="map_0", highest_order=13,
                ra_column="ra", dec_column="dec"), ra, dec


def test_sparse_checkpoint_matches_actual_healpix_counts_without_dense_arrays(tmp_path, monkeypatch):
    options, ra, dec = inputs(tmp_path)
    # At order 13 even one dense int64 sky array would need 6 GiB.
    def no_dense(*args, **kwargs):
        pytest.fail("Dense sky histogram allocated")

    monkeypatch.setattr("hats.pixel_math.sparse_histogram.HistogramAggregator.__init__", no_dense)
    result = _sparse_map_to_pixels(**options)
    saved = SparseHistogram.from_file(result["checkpoint"])
    pixels, counts = np.unique(hp.radec2pix(13, ra, dec), return_counts=True)
    np.testing.assert_array_equal(saved.indexes, pixels)
    np.testing.assert_array_equal(saved.counts, counts)
    assert result["rows"] == 5 and int(saved.order) == 13
    assert not list(tmp_path.glob("*.part"))


def test_mapping_failure_propagates_and_publishes_no_checkpoint(tmp_path):
    options, _, _ = inputs(tmp_path)
    options["ra_column"] = "absent"
    with pytest.raises(Exception):
        _sparse_map_to_pixels(**options)
    assert not list(tmp_path.rglob("*.npz"))


@pytest.mark.parametrize("processes", [False, True])
def test_actual_dask_retry_reexecutes_side_effects(tmp_path, processes):
    options, _, _ = inputs(tmp_path)
    with Client(processes=processes, n_workers=1, threads_per_worker=1,
                dashboard_address=None) as client:
        wrapped = _MappingClient(client)
        first = wrapped.submit(map_to_pixels, **options)
        result = first.result()
        from pathlib import Path
        target = Path(result["checkpoint"])
        target.unlink()
        second = wrapped.submit(map_to_pixels, **options)
        second.result()
        assert first.key != second.key
        assert target.is_file()
        assert int(SparseHistogram.from_file(str(target)).counts.sum()) == 5


def test_submission_window_checks_results_before_more_work(tmp_path):
    options, _, _ = inputs(tmp_path)
    class Future:
        key = "fake"
        def __init__(self, result):
            self.value = result
            self.checked = False
        def result(self, timeout):
            self.checked = True
            return self.value
    class Recorder:
        def __init__(self):
            self.futures = []
        def submit(self, function, *args, **kwargs):
            if len(self.futures) >= 3:
                assert self.futures[-3].checked
            future = Future(_sparse_map_to_pixels(*args, **{k: v for k, v in kwargs.items() if k != "pure"}))
            self.futures.append(future)
            return future
    recorder = Recorder()
    wrapped = _MappingClient(recorder, max_in_flight=2)
    for index in range(8):
        wrapped.submit(map_to_pixels, **(options | {"mapping_key": f"map_{index}"}))
        assert len(wrapped.pending) <= 2
    wrapped.finish_mapping()
    assert all(f.checked for f in recorder.futures)


def test_no_side_effect_result_fails_on_first_task():
    class Future:
        key = "no-op"
        def result(self, timeout):
            return None
    class Recorder:
        def submit(self, *args, **kwargs):
            return Future()
    with pytest.raises(RuntimeError, match="no verified checkpoint"):
        _MappingClient(Recorder()).submit(map_to_pixels)


def test_one_file_probe_cli(tmp_path):
    import subprocess
    import sys
    inputs(tmp_path)
    result = subprocess.run([
        sys.executable, "-m", "scripts.jades.probe_hats_mapping",
        "--scratch-dir", str(tmp_path), "--probe-root", str(tmp_path),
    ], text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "First HATS map verified from client:" in result.stdout
    assert "PASS: ONE mapping task executed" in result.stdout


def test_other_dask_tasks_are_not_replaced():
    class Recorder:
        def submit(self, function, *args, **kwargs):
            return function, args, kwargs
    function = lambda x: x
    assert _MappingClient(Recorder()).submit(function, 5) == (function, (5,), {})


def test_real_collection_import_keeps_nested_images_and_rows(tmp_path):
    from pathlib import Path
    directory = tmp_path / "inputs"
    directory.mkdir()
    images = [[[1.0, 2.0]], [[3.0, 4.0]]]
    pq.write_table(pa.table({"object_id": ["a", "b"], "ra": [53.0, 53.0001],
                             "dec": [-27.0, -27.0], "image": images}),
                   directory / "test.parquet")
    result = write_hats_from_parquet_dir(
        str(directory), str(tmp_path / "out"), "tiny", lowest_healpix_order=0,
        highest_healpix_order=2, pixel_threshold=100, expected_total_rows=2,
        tmp_dir=str(tmp_path / "tmp"), resumable=True, debug=True)
    leaves = sorted((Path(result) / "dataset").rglob("Npix=*.parquet"))
    rows = pa.concat_tables([pq.ParquetFile(p).read() for p in leaves]).to_pylist()
    assert {row["object_id"]: row["image"] for row in rows} == dict(zip(["a", "b"], images))
    assert (Path(result) / "hats.properties").is_file()
