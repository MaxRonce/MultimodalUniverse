# LSST DP2 image ingestion

This pipeline mirrors complete Rubin DP2 `deep_coadd` exposures for the
patches intersecting a parent Object catalog, then creates fixed-size MMU image
rows and a HATS catalog.

## Output contract

Each source has a six-band `image` struct in `u,g,r,i,z,y` order:

| Field | Shape/type | Meaning |
|---|---|---|
| `flux` | `(6,160,160) float32` | calibrated sky flux in nJy |
| `ivar` | `(6,160,160) float32` | inverse variance in nJy^-2 |
| `mask` | `(6,160,160) bool` | `True` only for usable, finite pixels |
| `mask_bits` | `(6,160,160) int32` | unmodified Rubin mask bit field |
| `psf_fwhm` | `(6,) float32` | arcsec; Object moments, then SIA fallback |
| `scale` | `(6,) float32` | 0.2 arcsec/pixel |
| `band_present` | `(6,) bool` | whether the patch-band was mirrored |

The default clean-pixel policy rejects `BAD`, `SAT`, `NO_DATA`, `SUSPECT`,
and `UNMASKEDNAN`. The original bits and per-file mask-plane mapping remain in
the row, so a different policy can be reconstructed without downloading DP2
again.

The MMU image schema represents the PSF as one FWHM per band. The complete
Rubin exposure mirrored on disk retains the serialized spatial PSF and native
provenance; producing per-source PSF pixel stamps would require a separate
LSST-stack transformation and is not part of this catalog schema.

## 1. Query the parent sample

Install the API dependencies and expose a Rubin token with `read:image` and
catalog access. Do not put the token in a file or command-line argument.

```bash
uv pip install pyvo requests
export RSP_TOKEN='...'

uv run python -m scripts.lsst_dp2.query_catalog \
  --ra 53.1246023 --dec -27.7404715 --radius-deg 0.05 \
  --where "refExtendedness = 1 AND detect_isIsolated = 1 AND i_cModelFlux / i_cModelFluxErr >= 10" \
  --output "$WORK/lsst_dp2/catalog/objects.parquet"
```

Here `refExtendedness = 1` selects extended sources (galaxy candidates), while
the `i`-band S/N cut avoids filling a pilot sample with threshold detections.
This is a morphological selection, not a spectroscopically confirmed label.

For a polygon, replace the cone arguments with, for example:

```bash
--polygon "53.0,-28.0 53.3,-28.0 53.3,-27.7 53.0,-27.7"
```

The exact ADQL is saved beside the catalog as `objects.parquet.adql`.

### Four-source authenticated smoke

From the repository root, create an isolated work area and provide a Rubin API
token without storing it in the repository:

```bash
cd /path/to/MultimodalUniverse
uv pip install --python .venv/bin/python pyvo requests

export RSP_TOKEN='...'
export LSST_SMOKE="$PWD/.local/lsst_dp2_smoke4"
mkdir -p "$LSST_SMOKE/catalog"
```

Query exactly four Object rows in one small sky region:

```bash
.venv/bin/python -m scripts.lsst_dp2.query_catalog \
  --ra 53.1246023 --dec -27.7404715 --radius-deg 0.02 \
  --limit 4 \
  --output "$LSST_SMOKE/catalog/objects.parquet"
```

Inspect the saved ADQL and verify the row count before downloading images:

```bash
cat "$LSST_SMOKE/catalog/objects.parquet.adql"
.venv/bin/python -c \
  "import pyarrow.parquet as p; print(p.read_metadata('$LSST_SMOKE/catalog/objects.parquet').num_rows)"
.venv/bin/python -c \
  "import pyarrow.parquet as p; t=p.read_table('$LSST_SMOKE/catalog/objects.parquet', columns=['tract','patch']); print(sorted(set(zip(t['tract'].to_pylist(), t['patch'].to_pylist()))))"
```

Mirror the complete six-band patch products. Four sources produce four MMU
cutouts, while each unique patch requires six network products (`ugrizy`):

```bash
.venv/bin/python -m scripts.lsst_dp2.download_coadds \
  --catalog "$LSST_SMOKE/catalog/objects.parquet" \
  --mirror-root "$LSST_SMOKE/coadds" \
  --manifest "$LSST_SMOKE/download_manifest.sqlite" \
  --workers 2
```

Check that all `6 * number_of_unique_patches` tasks completed:

```bash
.venv/bin/python -m scripts.lsst_dp2.download_coadds \
  --catalog "$LSST_SMOKE/catalog/objects.parquet" \
  --mirror-root "$LSST_SMOKE/coadds" \
  --manifest "$LSST_SMOKE/download_manifest.sqlite" \
  --status-only
```

Build and ingest the four rows:

```bash
.venv/bin/python -m scripts.lsst_dp2.build_parent_sample_hats \
  --catalog "$LSST_SMOKE/catalog/objects.parquet" \
  --manifest "$LSST_SMOKE/download_manifest.sqlite" \
  --scratch-dir "$LSST_SMOKE/parquet" \
  --output-root "$LSST_SMOKE/hats/lsst_dp2" \
  --objects-per-shard 2 \
  --pixel-threshold 32 \
  --ingest-workers 1 \
  --verify-checksums \
  --require-all-bands
```

The build must report four rows. Confirm the restart shards and HATS metadata:

```bash
.venv/bin/python -c \
  "from pathlib import Path; import pyarrow.parquet as p; q=Path('$LSST_SMOKE/parquet'); print(sum(p.read_metadata(x).num_rows for x in q.glob('part-*.parquet')))"
find "$LSST_SMOKE/hats" -name hats.properties -print
```

Run the fail-closed scientific validation gate:

```bash
.venv/bin/python -m scripts.lsst_dp2.validate_parent_sample \
  --catalog "$LSST_SMOKE/catalog/objects.parquet" \
  --manifest "$LSST_SMOKE/download_manifest.sqlite" \
  --scratch-dir "$LSST_SMOKE/parquet" \
  --hats-root "$LSST_SMOKE/hats/lsst_dp2" \
  --verify-checksums
```

This checks exact object identity, WCS centering within half a pixel on each
axis, native FITS units, six-band shapes, finite flux/ivar, mask consistency,
PSF and dataset provenance, and equal catalog/Parquet/HATS row counts. The
machine-readable receipt is `parquet/validation_report.json`.

## 2. Mirror complete useful patches

Run this from Jean-Zay or another host allowed to access the RSP API. Start at
low concurrency and set `--workers` from the live RSP quota page. Re-running
the same command retries failed manifest rows and skips completed checksummed
files.

```bash
uv run --no-sync python -m scripts.lsst_dp2.download_coadds \
  --catalog "$WORK/lsst_dp2/catalog/objects.parquet" \
  --mirror-root "$WORK/lsst_dp2/coadds" \
  --manifest "$WORK/lsst_dp2/download_manifest.sqlite" \
  --workers 4
```

Use `--limit 6` for the first one-patch smoke. The downloader follows the SIA
DataLink `#this` record (not a geometrically approximated SODA cutout). A product
is written as `tract=<id>/patch=<id>/band=<band>/deep_coadd.fits` only after FITS
validation, `fsync`, checksum, and atomic rename.

Inspect progress and the latest failures without requiring a Rubin token:

```bash
uv run --no-sync python -m scripts.lsst_dp2.download_coadds \
  --catalog "$WORK/lsst_dp2/catalog/objects.parquet" \
  --mirror-root "$WORK/lsst_dp2/coadds" \
  --manifest "$WORK/lsst_dp2/download_manifest.sqlite" \
  --status-only
```

## 3. Build restartable Parquet shards

For a smoke build:

```bash
uv run python -m scripts.lsst_dp2.build_parent_sample_hats \
  --catalog "$WORK/lsst_dp2/catalog/objects.parquet" \
  --manifest "$WORK/lsst_dp2/download_manifest.sqlite" \
  --scratch-dir "$WORK/lsst_dp2/parquet" \
  --output-root "$WORK/mmu_hats/lsst_dp2" \
  --verify-checksums
```

For a scatter/gather production build, submit independent shard jobs:

```bash
for i in $(seq 0 63); do
  sbatch --export=ALL,SHARD_IDX="$i" scripts/lsst_dp2/build_shard.slurm
done
```

Each shard command must include:

```bash
--num-shards 64 --shard-idx "$SHARD_IDX" --skip-ingest
```

After all 64 jobs succeed, run one gather job with the same catalog, manifest,
scratch and output paths plus `--only-ingest`. Never run gather while scatter
jobs are still writing Parquet files.

## Scaling

A six-band 3400x3400 patch requires about 0.83 GB just for float32 image and
variance plus int32 mask planes, before the additional ExposureF extensions and
compression. The MMU row payload is about 2.0 MB/source uncompressed when
including flux, ivar, boolean clean mask, and raw int32 mask bits; 5,000 sources
are therefore about 10 GB before Parquet/Zstandard compression. Network volume
scales with unique patches, while final HATS volume scales with source count.

### 5,000-galaxy Jean-Zay pilot

Jean-Zay compute nodes do not have Internet access. Run the TAP query and SIA
mirror from a login or pre/post-processing node with outbound access, then run
the restartable cutout build on CPU compute nodes. Keep data and caches under
`$WORK` or `$SCRATCH`, not `$HOME`.

Start with a compact region so the objects reuse a small number of patches:

```bash
export LSST_DP2_ROOT="$WORK/lsst_dp2_pilot5000"
mkdir -p "$LSST_DP2_ROOT/catalog"

uv run --no-sync python -m scripts.lsst_dp2.query_catalog \
  --ra 53.1246023 --dec -27.7404715 --radius-deg 0.20 \
  --where "refExtendedness = 1 AND detect_isIsolated = 1 AND i_cModelFlux / i_cModelFluxErr >= 10" \
  --limit 5000 \
  --output "$LSST_DP2_ROOT/catalog/objects.parquet"
```

Require exactly 5,000 unique IDs and inspect the number of unique patches,
which controls mirror volume:

```bash
uv run --no-sync python - <<'PY'
import os
from pathlib import Path
import pyarrow.parquet as pq

p = Path(os.environ["LSST_DP2_ROOT"]) / "catalog/objects.parquet"
t = pq.read_table(p, columns=["objectId", "tract", "patch"])
ids = t["objectId"].to_pylist()
patches = set(zip(t["tract"].to_pylist(), t["patch"].to_pylist()))
assert len(ids) == 5000 and len(set(ids)) == 5000
print(f"objects={len(ids)} unique_patches={len(patches)} download_tasks={6 * len(patches)}")
PY
```

Mirror from the connected node; rerunning this command resumes the manifest:

```bash
uv run --no-sync python -u -m scripts.lsst_dp2.download_coadds \
  --catalog "$LSST_DP2_ROOT/catalog/objects.parquet" \
  --mirror-root "$LSST_DP2_ROOT/coadds" \
  --manifest "$LSST_DP2_ROOT/download_manifest.sqlite" \
  --workers 4
```

For this pilot, use 16 scatter jobs. Run the gather only after every scatter
job has completed successfully:

```bash
mkdir -p logs
for i in $(seq 0 15); do
  sbatch --export=ALL,LSST_DP2_ROOT="$LSST_DP2_ROOT",NUM_SHARDS=16,SHARD_IDX="$i" \
    scripts/lsst_dp2/build_shard.slurm
done
```

After `sacct` shows that all 16 jobs completed with exit code `0:0`, submit the
single HATS gather and fail-closed validation job:

```bash
sbatch --export=ALL,LSST_DP2_ROOT="$LSST_DP2_ROOT" \
  scripts/lsst_dp2/gather_validate.slurm
```

The pilot is complete only when this job exits `0:0` and
`$LSST_DP2_ROOT/parquet/validation_report.json` contains `"status": "PASS"`.

## Required validation

1. Compare one patch-band against Butler for flux, variance, integer mask, WCS,
   dimensions and pixel scale.
2. Assert six channels, finite flux/ivar, non-negative ivar, and
   `ivar[~mask] == 0` after HATS read-back.
3. Confirm catalog row count equals HATS row count and object IDs are unique.
4. Inspect the SQLite manifest for failed or incomplete tasks before claiming
   production completion.
