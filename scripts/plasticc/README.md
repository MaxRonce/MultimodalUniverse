# PLAsTiCC Jagged HATS Builder

`build_parent_sample_hats.py` replaces the legacy padded HDF5 conversion for MMU
v2. It reads the official train and optional unblinded-test CSV files, preserves
every raw observation in a jagged `lightcurve` struct, and writes quality and
detection masks separately.

```text
lightcurve: struct<
  time: list<float64>,
  band: list<string>,
  band_id: list<int16>,
  flux: list<float32>,
  flux_err: list<float32>,
  finite_mask: list<bool>,
  quality_mask: list<bool>,
  detected_mask: list<bool>
>
```

The default `plasticc_aion_v1` object policy requires at least 30 quality epochs
in at least two bands. It is recorded as `aion_training_candidate_v1`, the
compatibility alias `aion_training_candidate`, and `quality_policy_version`.

```bash
python -m scripts.plasticc.build_parent_sample_hats \
  --raw-root /path/to/plasticc_csvs \
  --scratch-dir /path/to/plasticc_parquet_scratch \
  --output-root /path/to/mmu_v2 \
  --resume
```

For an array build, use a shared scratch directory with `--num-shards 12`, one
distinct `--shard-idx` per task, and `--skip-ingest`. After the array succeeds,
run once with `--only-ingest` against the same scratch directory.
