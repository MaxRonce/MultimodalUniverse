# LSST DP2 images in MMU

This pipeline builds an MMU HATS image catalog from Rubin LSST DP2
`deep_coadd` products. It mirrors each useful complete `(tract, patch, band)`
FITS once, then extracts all source cutouts locally.

The implementation is split by responsibility:

- `query_catalog.py`: authenticated DP2 Object TAP query;
- `download_coadds.py`: resumable SIA/DataLink mirror with a SQLite manifest;
- `coadd.py`: calibrated FITS, mask, WCS, and local cell-PSF extraction;
- `schema.py`: nested Arrow representation of the MMU image contract;
- `build_parent_sample_hats.py`: restartable patch processing and HATS ingest;
- `validate_parent_sample.py`: fail-closed catalog-to-HATS validation;
- `Snakefile`: single-node Jean-Zay workflow with durable stage markers.

## Data contract

Each source has a fixed `160 x 160` cutout centered from its DP2 Object
`coord_ra`, `coord_dec`. Band order is always `u, g, r, i, z, y`.

The `image` struct contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `flux` | `(6,160,160)` float32 | calibrated coadd surface pixels in `nJy` |
| `ivar` | `(6,160,160)` float32 | inverse variance in `nJy^-2` |
| `mask` | `(6,160,160)` bool | valid pixel after Rubin mask rejection |
| `mask_bits` | `(6,160,160)` int32 | original Rubin integer bit mask |
| `psf_image` | `(6,35,35)` float32 | normalized local DP2 cell PSF |
| `psf_image_valid` | `(6,)` bool | whether the local PSF cell is usable |
| `psf_fwhm` | `(6,)` float32 | catalog/SIA scalar PSF summary in arcsec |
| `scale` | `(6,)` float32 | pixel scale, currently `0.2` arcsec/pixel |
| `band_present` | `(6,)` bool | availability of each deep coadd |
| provenance | `(6,)` | dataset ID, SHA-256, mask-plane map, PSF source |

Invalid pixels have `ivar=0` and `mask=False`. Non-finite input flux is stored
as zero only where the mask rejects that pixel. A missing or non-finite Rubin
PSF cell is stored as a zero kernel with `psf_image_valid=False`; it is never
replaced with a neighboring PSF.

The default rejected mask planes are `BAD,SAT,NO_DATA,SUSPECT,UNMASKEDNAN`.
The raw mask bits and their plane mapping remain in the output, so downstream
users can define a different policy.

## Local verification

From the repository root:

```bash
uv sync --frozen --extra dev --extra viz
uv run pytest -q tests/test_lsst_dp2_build.py tests/test_hats_configs.py
uvx ruff check scripts/lsst_dp2 tests/test_lsst_dp2_build.py
```

The tests use synthetic FITS files and do not contact Rubin services.

## Jean-Zay installation

Everything below, including the repository, virtual environment, caches,
temporary files, logs, coadds, Parquet, and HATS output, lives under `$SCRATCH`.
Run the setup on a `prepost` allocation, not on a login node.

```bash
srun \
  --account=jrx@cpu \
  --partition=prepost \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --time=20:00:00 \
  --pty bash -i
```

Inside the allocation:

```bash
set -euo pipefail

export MMU_JZ_ROOT="$SCRATCH/mmu_lsst_dp2"
mkdir -p "$MMU_JZ_ROOT"
cd "$MMU_JZ_ROOT"

# First installation only. For an existing checkout, use git pull --ff-only.
git clone --branch feat/lsst-dp2 \
  https://github.com/MaxRonce/MultimodalUniverse.git
cd MultimodalUniverse

export LSST_DP2_RUN_NAME="pilot5000_clean"
source scripts/lsst_dp2/jeanzay_env.sh

# Bootstrap uv itself under SCRATCH when it is not already available.
if ! command -v uv >/dev/null 2>&1; then
  python3 -m venv "$MMU_JZ_ROOT/venvs/uv"
  "$MMU_JZ_ROOT/venvs/uv/bin/pip" install --upgrade pip uv
  export PATH="$MMU_JZ_ROOT/venvs/uv/bin:$PATH"
fi

uv sync --frozen --extra dev --extra viz
"$MMU_PYTHON" -c \
  "import mmu, astropy, pyarrow, pyvo, hats, lsdb; print('installation OK')"
```

On subsequent sessions, restore the same environment with:

```bash
export MMU_JZ_ROOT="$SCRATCH/mmu_lsst_dp2"
export LSST_DP2_RUN_NAME="pilot5000_clean"
cd "$MMU_JZ_ROOT/MultimodalUniverse"
source scripts/lsst_dp2/jeanzay_env.sh
```

## Build a 5,000-galaxy validation sample

### 1. Authenticate

Create a fresh token at the Rubin Science Platform. Do not write it to a file
or place it directly in shell history.

```bash
read -rsp "RSP token: " RSP_TOKEN
echo
export RSP_TOKEN
```

### 2. Query the catalog

This validation sample selects extended sources but does not impose S/N or
isolation cuts. `--limit` queries are ordered by `objectId`, so rebuilding the
same cone is deterministic. The saved `.adql` file records the exact query.

```bash
"$MMU_PYTHON" -u -m scripts.lsst_dp2.query_catalog \
  --ra 53.1246023 \
  --dec -27.7404715 \
  --radius-deg 0.20 \
  --where "refExtendedness = 1" \
  --limit 5000 \
  --output "$LSST_DP2_ROOT/catalog/objects.parquet"
```

This selection is only a validation pilot. Production must enumerate the
desired DP2 footprint without S/N, isolation, or morphology cuts and preserve
the corresponding catalog metadata for downstream selections.

### 3. Check the query result

Do not continue if the cone contains fewer than 5,000 extended objects.

```bash
"$MMU_PYTHON" - <<'PY'
import os
from pathlib import Path
import pyarrow.parquet as pq

path = Path(os.environ["LSST_DP2_ROOT"]) / "catalog/objects.parquet"
table = pq.read_table(path, columns=["objectId", "tract", "patch"])
ids = table["objectId"].to_pylist()
patches = set(zip(table["tract"].to_pylist(), table["patch"].to_pylist()))
assert len(ids) == 5000, f"expected 5000 rows, found {len(ids)}"
assert len(ids) == len(set(ids)), "duplicate objectId"
print(f"objects={len(ids)} patches={len(patches)} download_tasks={6 * len(patches)}")
PY
```

If it returns fewer rows, increase `--radius-deg` and use a new
`LSST_DP2_RUN_NAME`. A manifest is deliberately bound to one exact catalog and
mirror root.

### 4. Run the complete workflow

Inside the interactive allocation:

```bash
bash scripts/lsst_dp2/run_prepost.sh
```

The DAG performs, in order:

1. resumable SIA downloads with four workers;
2. patch-local cutout extraction into restartable Parquet shards;
3. HATS ingestion with four Dask workers;
4. checksum, WCS, centering, unit, shape, mask, ivar, PSF, provenance, row-count,
   and HATS metadata validation.

Transient Rubin/proxy failures are retried by Snakemake. Running the same
command again resumes complete downloads and Parquet files.

For a launch that survives terminal disconnection, leave the interactive
allocation after installation and catalog creation. Exports made inside
`srun` do not propagate back to the login shell, so restore them there before
submitting:

```bash
export MMU_JZ_ROOT="$SCRATCH/mmu_lsst_dp2"
export LSST_DP2_RUN_NAME="pilot5000_clean"
cd "$MMU_JZ_ROOT/MultimodalUniverse"
source scripts/lsst_dp2/jeanzay_env.sh
read -rsp "RSP token: " RSP_TOKEN
echo
export RSP_TOKEN

mkdir -p "$LSST_DP2_ROOT/logs"
JOB_ID=$(sbatch --parsable \
  --account=jrx@cpu \
  --partition=prepost \
  --export=ALL \
  --output="$LSST_DP2_ROOT/logs/workflow-%j.out" \
  --error="$LSST_DP2_ROOT/logs/workflow-%j.err" \
  scripts/lsst_dp2/pilot_prepost.slurm)
echo "$JOB_ID"
```

### 5. Monitor

```bash
squeue -j "$JOB_ID"
tail -F "$LSST_DP2_ROOT/logs/workflow-$JOB_ID.out"
tail -F "$LSST_DP2_ROOT/logs/download.log"
tail -F "$LSST_DP2_ROOT/logs/build_parquet.log"
tail -F "$LSST_DP2_ROOT/logs/ingest_hats.log"
tail -F "$LSST_DP2_ROOT/logs/validate.log"
```

After the job leaves `squeue`, check terminal state rather than assuming that
disappearance means success:

```bash
sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

### 6. Acceptance gate

```bash
"$MMU_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["LSST_DP2_ROOT"])
report = json.loads((root / "validation_report.json").read_text())
assert report["status"] == "PASS"
assert report["catalog_rows"] == report["parquet_rows"] == report["hats_rows"] == 5000
assert report["center_checks"] == 6 * 5000
print(json.dumps({
    "status": report["status"],
    "rows": report["hats_rows"],
    "coadds": report["coadd_products"],
    "max_center_offset_pix": report["max_center_radial_offset_pix"],
    "psf_valid_fractions": report["psf_valid_fractions"],
    "elapsed_seconds": report["elapsed_seconds"],
}, indent=2))
PY
```

The final catalog is below:

```text
$MMU_HATS_ROOT/lsst_dp2/lsst_dp2/lsst_dp2/
```

## Recovery

Inspect the download manifest without a token:

```bash
"$MMU_PYTHON" -m scripts.lsst_dp2.download_coadds \
  --catalog "$LSST_DP2_ROOT/catalog/objects.parquet" \
  --mirror-root "$LSST_DP2_ROOT/coadds" \
  --manifest "$LSST_DP2_ROOT/download_manifest.sqlite" \
  --status-only
```

If tasks reached the attempt limit after a resolved external outage:

```bash
"$MMU_PYTHON" -m scripts.lsst_dp2.download_coadds \
  --catalog "$LSST_DP2_ROOT/catalog/objects.parquet" \
  --mirror-root "$LSST_DP2_ROOT/coadds" \
  --manifest "$LSST_DP2_ROOT/download_manifest.sqlite" \
  --reset-failed --workers 2
```

Do not delete individual Parquet files or edit the SQLite database manually.
If the catalog, mask policy, schema, or processing code changes, start a new
`LSST_DP2_RUN_NAME`. The build contract intentionally refuses to mix products
from different inputs or code versions.

## Scientific release gate

A technical `PASS` proves internal consistency, not absence of scientific
bias. Before a release or full DP2 production run, compare a fixed independent
sample against Butler-generated cutouts for:

- flux, variance, integer mask, WCS, dimensions, and units;
- catalog aperture/PSF photometry reconstructed from the cutouts;
- normalized background residuals and resampling-induced covariance;
- local PSF kernel selection and invalid-cell frequency;
- truncation and edge effects as a function of size, brightness, and position.

Record that comparison, the git commit, saved ADQL, manifest identity, build
contract, validation report, Slurm receipt, and final HATS row count as the
dataset provenance bundle.
