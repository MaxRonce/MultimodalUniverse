# ZTF DR23 HATS Conversion

This builder converts the public ZTF DR23 light-curve HATS product from the
IPAC/ZTF schema into the MMU HATS time-series schema.

The input is already HATS. The conversion preserves its native row unit: one
output row is one ZTF light-curve-series row. `objectid` is kept as
`ztf_objectid` and exposed as string `object_id`, but it is not treated as a
validated cross-band astrophysical-object identifier. No rows or epochs are
filtered during conversion.

Output light curves are jagged, not padded:

```text
lightcurve: struct<
  time: list<float64>,        # ZTF hmjd
  band: list<string>,         # ztf_g / ztf_r / ztf_i, repeated per epoch
  band_id: list<int16>,       # raw filterid per epoch
  mag: list<float32>,
  mag_err: list<float32>,
  clrcoeff: list<float32>,
  catflags: list<int32>,
  finite_mask: list<bool>,
  poserr_mask: list<bool>,
  relaxed_mask: list<bool>,
  strict_mask: list<bool>,
  bright_mask: list<bool>
>
```

Top-level counters and flags include:

```text
n_total, n_finite, n_poserr, n_relaxed, n_strict, n_bright_relaxed
pass_relaxed_ge3/ge5/ge10/ge20/ge30/ge50
pass_strict_ge3/ge5/ge10/ge20/ge30/ge50
aion_training_candidate
aion_training_candidate_v1
quality_policy_version = ztf_aion_relaxed_ge30_v1
```

By default, `aion_training_candidate = aion_training_candidate_v1 =
pass_relaxed_ge30`. The versioned name and `quality_policy_version` prevent a
future threshold change from silently changing the meaning of saved training
views.

Quality definitions:

```text
finite_epoch  = isfinite(hmjd) & isfinite(mag) & isfinite(magerr) & filterid in {1,2,3}
poserr_epoch  = finite_epoch & magerr > 0
relaxed_epoch = poserr_epoch & ((catflags & 32768) == 0)
strict_epoch  = relaxed_epoch & (catflags == 0)
bright_epoch  = relaxed_epoch & mag < 15.5
```

Direct HATS smoke test on Jean-Zay:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/ZTF/MultimodalUniverse

SUBSET="/lustre/fsn1/projects/rech/jrx/urx63nr/ztf_dr23_lc_hats_subset_10GB/contributed/dr23/lc/hats/ztf_dr23_lc-hats"
OUT="/lustre/fswork/projects/rech/jrx/urx63nr/ZTF/mmu_ztf_dr23_subset_hats_direct"
MANIFEST="/lustre/fswork/projects/rech/jrx/urx63nr/ZTF/mmu_ztf_dr23_subset_hats_direct_manifests"

python -m scripts.ztf.build_parent_sample_hats \
  --input-dir "$SUBSET" \
  --output-root "$OUT" \
  --scratch-dir "$MANIFEST" \
  --direct-hats \
  --num-processes 8 \
  --batch-size 10000 \
  --resume
```

For full production, use the direct HATS mode rather than the legacy
scratch-plus-`hats_import` gather. Each array task converts a stride slice of
the input HATS parquet files into the same relative HATS partition path under
the output catalog. After all shards complete, run one metadata finalization
job:

```bash
INPUT="/lustre/fsn1/projects/rech/jrx/urx63nr/ztf_dr23_lc_hats_full/ztf_dr23_lc-hats"
OUT="/lustre/fswork/projects/rech/jrx/urx63nr/ZTF/mmu_ztf_dr23_full_hats"
MANIFEST="/lustre/fswork/projects/rech/jrx/urx63nr/ZTF/mmu_ztf_dr23_full_hats_manifests"
SHARDS=512

# Scatter task body. In Slurm, set SHARD_IDX from SLURM_ARRAY_TASK_ID.
python -u -m scripts.ztf.build_parent_sample_hats \
  --input-dir "$INPUT" \
  --output-root "$OUT" \
  --scratch-dir "$MANIFEST" \
  --direct-hats \
  --num-shards "$SHARDS" \
  --shard-idx "$SHARD_IDX" \
  --num-processes 8 \
  --batch-size 10000 \
  --resume \
  --skip-ingest

# Finalize once after all shards are done.
python -u -m scripts.ztf.build_parent_sample_hats \
  --input-dir "$INPUT" \
  --output-root "$OUT" \
  --scratch-dir "$MANIFEST" \
  --finalize-direct-hats \
  --num-shards "$SHARDS" \
  --shard-idx 0
```

The direct path keeps every row and epoch, preserves `_healpix_29`, writes
`ztf/ztf/dataset/Norder=.../Dir=.../Npix=....parquet`, and finalizes
`collection.properties`, `hats.properties`, `partition_info.csv`, and parquet
metadata files without a global spatial repartition.

On Jean-Zay, submit `build_full_jeanzay.slurm` from the MMU repository root,
then submit `finalize_full_jeanzay.slurm` with an `afterok` dependency on the
array job. The default output is already
`$SCRATCH/TimeSeries/local/ztf/ztf/ztf`, so the 8 TB catalog is not copied a
second time merely to change its parent directory.
