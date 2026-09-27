# LSST DP2 images in MMU

This pipeline builds an MMU HATS image catalog from Rubin LSST DP2
`deep_coadd` products. It mirrors each useful complete `(tract, patch, band)`
FITS once, then extracts all source cutouts locally.

The implementation is split by responsibility:

- `query_catalog.py`: authenticated DP2 Object TAP query;
- `query_multiregion_catalog.py`: balanced deterministic multi-region query;
- `query_dense_patch_catalog.py`: ranked patch selection and disjoint extensions;
- `submit_dense_extension.sh`: sequential download allocations for an extension;
- `stratify_catalog.py`: deterministic magnitude-size sampling;
- `download_coadds.py`: resumable SIA/DataLink mirror with a SQLite manifest;
- `coadd.py`: calibrated FITS, mask, WCS, and local cell-PSF extraction;
- `schema.py`: nested Arrow representation of the MMU image contract;
- `build_parent_sample_hats.py`: restartable patch processing and HATS ingest;
- `validate_parent_sample.py`: fail-closed catalog-to-HATS validation;
- `plot_mag_size_gallery.py`: auditable per-cell RGB galleries;
- `plot_mag_size_band_gallery.py`: full `160 x 160` grayscale `ugrizy` panels;
- `review_dataset.py`: final HATS count checks, random QA panels, and raw NPZs;
- `publish_review.py`: explicit W&B upload of review panels and metadata;
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
| `psf_fwhm` | `(6,)` float32 | catalog/SIA scalar PSF summary in arcsec; zero when unavailable |
| `scale` | `(6,)` float32 | pixel scale, currently `0.2` arcsec/pixel |
| `band_present` | `(6,)` bool | availability of each deep coadd |
| provenance | `(6,)` | dataset ID, SHA-256, mask-plane map, PSF source |

Invalid pixels have `ivar=0` and `mask=False`. Non-finite input flux is stored
as zero only where the mask rejects that pixel. A missing or non-finite Rubin
PSF cell is stored as a zero kernel with `psf_image_valid=False`; it is never
replaced with a neighboring PSF. Sources outside a product's tabulated PSF grid
use the same explicit missing-PSF representation; their calibrated image,
inverse variance, masks, and scalar catalog PSF summary remain available.
An unavailable scalar FWHM is independently represented by `psf_fwhm=0` and
`psf_source="missing"`; validation reports its availability per band without
rejecting an otherwise valid cutout.

DP2 does not provide every band for every patch. A SIA query with no matching
product is a terminal `unavailable` manifest state, not a download failure. The
corresponding MMU band keeps its fixed `ugrizy` slot with `flux=0`, `ivar=0`,
`mask=False`, `mask_bits=0`, a zero PSF, and `band_present=False`. This is
explicit missingness; downstream code must use `band_present` and must not
interpret the zero-filled plane as an observed image.

The default rejected mask planes are `BAD,SAT,NO_DATA,SUSPECT,UNMASKEDNAN`.
The raw mask bits and their plane mapping remain in the output, so downstream
users can define a different policy.

## Local verification

From the repository root:

```bash
uv sync --frozen --extra dev --extra viz
uv run python -m pytest -q tests/test_lsst_dp2_build.py tests/test_hats_configs.py
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

1. resumable, rate-limited SIA downloads with four workers;
2. four-way patch-local cutout extraction into restartable Parquet shards;
3. memory-bounded HATS ingestion with two Dask workers;
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
assert report["center_checks"] == sum(report["band_present_counts"].values())
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
$LSST_DP2_ROOT/hats/lsst_dp2/lsst_dp2/lsst_dp2/
```

## Build a 50,000-object multi-region qualification sample

This is the next scale gate after the 628-object visual validation. It selects
10,000 candidate galaxies in each of five separated DP2 regions. Selection is
deterministic within each region and imposes no effective-radius cut:

```sql
i_cModelMag < 22
AND i_extendedness = 1
AND griz_model_extendedness >= 0.8
AND sersic_no_data_flag = 0
AND sersic_unknown_flag = 0
```

The five default regions are DDF ELAIS-S1, DDF ECDFS, DDF EDFS-a, DDF COSMOS,
and Rubin SV 225 -40. Their coordinates follow the
[official DP2 region list](https://dp2.lsst.io/tutorials/notebook/301/notebook-301-1.html).

### 1. Prepare the run on a prepost node

```bash
export MMU_JZ_ROOT="$SCRATCH/mmu_lsst_dp2"
export LSST_DP2_RUN_NAME="multiregion_i22_50k_v1"
cd "$MMU_JZ_ROOT/MultimodalUniverse"
source scripts/lsst_dp2/jeanzay_env.sh

read -rsp "RSP token: " RSP_TOKEN
echo
export RSP_TOKEN
```

All run products, including HATS, now live below `$LSST_DP2_ROOT`; a new run
cannot overwrite another run's catalog.

### 2. Query exactly 50,000 objects

```bash
"$MMU_PYTHON" -u -m scripts.lsst_dp2.query_multiregion_catalog \
  --per-region 10000 \
  --seed 20260908 \
  --output "$LSST_DP2_ROOT/catalog/objects.parquet"
```

This performs five asynchronous TAP queries, retains the full eligible pool in
memory one region at a time, and selects by a stable hash of `objectId`. The
companion `objects.parquet.selection.json` records every query, region count,
and coordinate. No image request is made at this stage.

### 3. Preflight rows, patches, and storage

```bash
"$MMU_PYTHON" - <<'PY'
import os
import shutil
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

root = Path(os.environ["LSST_DP2_ROOT"])
table = pq.read_table(
    root / "catalog/objects.parquet",
    columns=["objectId", "tract", "patch", "dp2_region"],
)
ids = table["objectId"].to_pylist()
patches = set(zip(table["tract"].to_pylist(), table["patch"].to_pylist()))
regions = Counter(table["dp2_region"].to_pylist())
free_gib = shutil.disk_usage(root).free / 1024**3

assert len(ids) == len(set(ids)) == 50000
assert set(regions.values()) == {10000}
print("regions:", dict(sorted(regions.items())))
print("unique patches:", len(patches))
print("download tasks:", 6 * len(patches))
print("estimated coadds GiB:", round(6 * len(patches) * 31 / 1024, 1))
print("filesystem free GiB:", round(free_gib, 1))
print("Require at least 350 GiB of usable project allocation for this pilot.")
PY
```

`df` reports filesystem capacity, not necessarily the user's project quota.
Confirm the applicable IDRIS quota before submission. The 350 GiB allowance
covers mirrored coadds, intermediate Parquet, final HATS, and restart margin.

### 4. Submit one prepost node

The defaults are four download workers, a global limit of 50 request starts per
minute, four concurrent patch builders, two HATS workers, and a 256-row HATS
partition threshold. The request limiter covers SIA, DataLink, and FITS GET
starts together.

```bash
mkdir -p "$LSST_DP2_ROOT/logs"
JOB_ID=$(sbatch --parsable \
  --account=jrx@cpu \
  --partition=prepost \
  --export=ALL \
  --output="$LSST_DP2_ROOT/logs/workflow-%j.out" \
  --error="$LSST_DP2_ROOT/logs/workflow-%j.err" \
  scripts/lsst_dp2/pilot_prepost.slurm)
echo "JOB_ID=$JOB_ID"
```

Do not start a second workflow against the same run directory. Monitor with:

```bash
squeue -j "$JOB_ID"
tail -F "$LSST_DP2_ROOT/logs/download.log"
tail -F "$LSST_DP2_ROOT/logs/build_parquet.log"
tail -F "$LSST_DP2_ROOT/logs/ingest_hats.log"
tail -F "$LSST_DP2_ROOT/logs/validate.log"
```

When the job ends:

```bash
sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

### 5. Acceptance gate

```bash
"$MMU_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["LSST_DP2_ROOT"])
report = json.loads((root / "validation_report.json").read_text())
assert report["status"] == "PASS"
assert report["catalog_rows"] == 50000
assert report["parquet_rows"] == 50000
assert report["hats_rows"] == 50000
assert report["unique_object_ids"] == 50000
assert report["center_checks"] == sum(report["band_present_counts"].values())
assert report["max_center_axis_offset_pix"] <= 0.500001
assert set(report["psf_valid_fractions"]) == set("ugrizy")
assert set(report["band_present_fractions"]) == set("ugrizy")
print(json.dumps(report, indent=2))
PY
```

The validated MMU catalog is:

```text
$LSST_DP2_ROOT/hats/lsst_dp2/lsst_dp2/lsst_dp2/
```

For a zero-to-result batch launch, `qualification_50k_prepost.slurm` performs
the environment installation, catalog query, preflight, workflow, and final
validation within one `prepost` allocation. The front node is used only to
clone/update the repository, read the token, and submit the job; see the launch
recipe below.

```bash
set -euo pipefail

export MMU_JZ_ROOT="$SCRATCH/mmu_lsst_dp2"
export MMU_REPO="$MMU_JZ_ROOT/MultimodalUniverse"
export LSST_DP2_RUN_NAME="multiregion_i22_50k_v1"
mkdir -p "$MMU_JZ_ROOT"

if [[ ! -d "$MMU_REPO/.git" ]]; then
  git clone --branch feat/lsst-dp2 \
    https://github.com/MaxRonce/MultimodalUniverse.git \
    "$MMU_REPO"
else
  git -C "$MMU_REPO" fetch origin feat/lsst-dp2
  git -C "$MMU_REPO" switch feat/lsst-dp2
  git -C "$MMU_REPO" pull --ff-only origin feat/lsst-dp2
fi

unset LSST_DP2_ROOT LSST_DP2_HATS_OUTER
source "$MMU_REPO/scripts/lsst_dp2/jeanzay_env.sh"
read -rsp "RSP token: " RSP_TOKEN
echo
export RSP_TOKEN

JOB_ID=$(sbatch --parsable \
  --account=jrx@cpu \
  --partition=prepost \
  --export=ALL \
  --output="$LSST_DP2_ROOT/logs/qualification-%j.out" \
  --error="$LSST_DP2_ROOT/logs/qualification-%j.err" \
  "$MMU_REPO/scripts/lsst_dp2/qualification_50k_prepost.slurm")
unset RSP_TOKEN
echo "JOB_ID=$JOB_ID"
```

The job can be resubmitted with the same run name after a timeout or transient
service failure. It reuses the exact catalog, completed downloads, and valid
intermediate shards. On startup, interrupted `running` tasks return to `pending`;
legacy zero-result SIA failures become terminal `unavailable` tasks. No completed
FITS product is downloaded again.

After pulling a pipeline update, resume the same run from a Jean-Zay front node:

```bash
export MMU_JZ_ROOT="$SCRATCH/mmu_lsst_dp2"
export MMU_REPO="$MMU_JZ_ROOT/MultimodalUniverse"
export LSST_DP2_RUN_NAME="multiregion_i22_50k_v1"

git -C "$MMU_REPO" fetch origin feat/lsst-dp2
git -C "$MMU_REPO" switch feat/lsst-dp2
git -C "$MMU_REPO" pull --ff-only origin feat/lsst-dp2

unset LSST_DP2_ROOT LSST_DP2_HATS_OUTER
source "$MMU_REPO/scripts/lsst_dp2/jeanzay_env.sh"
read -rsp "RSP token: " RSP_TOKEN
echo
export RSP_TOKEN

mkdir -p "$LSST_DP2_ROOT/logs"
JOB_ID=$(sbatch --parsable \
  --account=jrx@cpu \
  --partition=prepost \
  --export=ALL \
  --output="$LSST_DP2_ROOT/logs/qualification-%j.out" \
  --error="$LSST_DP2_ROOT/logs/qualification-%j.err" \
  "$MMU_REPO/scripts/lsst_dp2/qualification_50k_prepost.slurm")
unset RSP_TOKEN
echo "JOB_ID=$JOB_ID"
```

## Magnitude-size validation sample

The 5,000-row smoke test above checks the pipeline, but it is not a controlled
visual sample. Use the following workflow to compare morphology across
brightness and apparent size without selecting the most attractive or
highest-S/N objects.

The default grid has five magnitude bins and four major-axis effective-radius
bins. With 32 objects per cell it contains at most 640 galaxy candidates:

- `18 <= i < 20`, then one-magnitude bins through `i < 24`;
- `0.4 <= Re < 0.6`, `0.6 <= Re < 1.0`, `1.0 <= Re < 1.5`, and
  `Re >= 1.5` arcsec.

Start a fresh run so its manifest and products cannot be mixed with the
5,000-row smoke test:

```bash
export MMU_JZ_ROOT="$SCRATCH/mmu_lsst_dp2"
export LSST_DP2_RUN_NAME="ecdfs_mag_size_v1"
cd "$MMU_JZ_ROOT/MultimodalUniverse"
source scripts/lsst_dp2/jeanzay_env.sh

read -rsp "RSP token: " RSP_TOKEN
echo
export RSP_TOKEN
```

Query a parent candidate pool in the DP2 E-CDFS region. This query deliberately
keeps all candidates in the spatial region before the deterministic sampling
step:

```bash
"$MMU_PYTHON" -u -m scripts.lsst_dp2.query_catalog \
  --ra 53.0 \
  --dec -28.1 \
  --radius-deg 0.30 \
  --where "i_cModelMag >= 18 AND i_cModelMag < 24 AND i_extendedness = 1 AND griz_model_extendedness >= 0.8 AND sersic_no_data_flag = 0 AND sersic_unknown_flag = 0 AND sersic_reff_major >= 0.4" \
  --output "$LSST_DP2_ROOT/catalog/candidates.parquet"
```

Create the balanced sample. Selection inside each cell is based on a stable
hash of `objectId`, not on S/N, color, appearance, or catalog order:

```bash
"$MMU_PYTHON" -u -m scripts.lsst_dp2.stratify_catalog \
  --catalog "$LSST_DP2_ROOT/catalog/candidates.parquet" \
  --output "$LSST_DP2_ROOT/catalog/objects.parquet" \
  --mag-edges "18,20,21,22,23,24" \
  --size-edges "0.4,0.6,1.0,1.5,inf" \
  --per-cell 32 \
  --seed 20260908 \
  --require-full-cells
```

The companion file `objects.parquet.selection.json` records availability and
selected counts for every cell. If a cell is underfilled, the command fails
before writing the sample; enlarge the cone or reduce `--per-cell` explicitly.

Build and validate the cutouts with the same restartable workflow:

```bash
bash scripts/lsst_dp2/run_prepost.sh
```

Check the actual expected row count rather than assuming 640 rows:

```bash
"$MMU_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["LSST_DP2_ROOT"])
selection = json.loads(
    (root / "catalog/objects.parquet.selection.json").read_text()
)
validation = json.loads((root / "validation_report.json").read_text())
assert validation["status"] == "PASS"
assert validation["catalog_rows"] == selection["selected_rows"]
assert validation["parquet_rows"] == selection["selected_rows"]
assert validation["hats_rows"] == selection["selected_rows"]
print(f"PASS: {selection['selected_rows']} cutouts")
PY
```

Generate two gallery sets. The first uses one fixed display stretch across all
cells and is the appropriate comparison plot. The second adapts the display
stretch independently to reveal faint morphology. Neither changes the stored
MMU flux, ivar, mask, or PSF arrays.

```bash
export HATS_PATH="$LSST_DP2_ROOT/hats/lsst_dp2/lsst_dp2/lsst_dp2"

"$MMU_PYTHON" -u -m scripts.lsst_dp2.plot_mag_size_gallery \
  --hats-path "$HATS_PATH" \
  --output-dir "$LSST_DP2_ROOT/figures/fixed" \
  --mag-edges "18,20,21,22,23,24" \
  --size-edges "0.4,0.6,1.0,1.5,inf" \
  --per-cell 16 \
  --stretch-njy 10 \
  --smooth-sigma 0 \
  --display-sigma 0

"$MMU_PYTHON" -u -m scripts.lsst_dp2.plot_mag_size_gallery \
  --hats-path "$HATS_PATH" \
  --output-dir "$LSST_DP2_ROOT/figures/adaptive" \
  --mag-edges "18,20,21,22,23,24" \
  --size-edges "0.4,0.6,1.0,1.5,inf" \
  --per-cell 16
```

Each directory contains one PNG per populated cell, a CSV listing every shown
`objectId`, magnitude, and size, and a JSON file recording the rendering
parameters. This E-CDFS run is a controlled deep-field diagnostic, not a
sky-representative DP2 sample; repeat the same grid in other DP2 regions before
using the result to characterize survey-wide population diversity.

For a view closest to the stored MMU product, render all six bands separately
at the full cutout size. This command applies no crop, smoothing, or background
subtraction. The asinh operation is display normalization only and its settings
are recorded in `rendering.json`:

```bash
"$MMU_PYTHON" -u -m scripts.lsst_dp2.plot_mag_size_band_gallery \
  --hats-path "$HATS_PATH" \
  --output-dir "$LSST_DP2_ROOT/figures/full_bands" \
  --mag-edges "18,20,21,22,23,24" \
  --size-edges "0.4,0.6,1.0,1.5,inf" \
  --per-cell 8 \
  --stretch asinh \
  --shared-object-scale
```

Build a self-contained HTML viewer with one slider for magnitude and one for
effective radius from the 20 full-band panels:

```bash
"$MMU_PYTHON" -m scripts.lsst_dp2.build_interactive_gallery \
  --input-dir "$LSST_DP2_ROOT/figures/full_bands" \
  --output "$LSST_DP2_ROOT/figures/interactive_mag_re.html"
```

### Photometric redshifts

Redshifts are not columns of the TAP `dp2.Object` table. DP2 also provides a
separate provisional HATS catalog at `/rubin/lsdb_data/dp2/object_photoz` in
the Rubin Science Platform environment. It contains estimates from multiple
photo-z algorithms and uncertainty intervals and can be joined to the selected
sample by `objectId`. Preserve the algorithm name, point estimate, and interval
bounds; do not reduce the photo-z information to one undocumented scalar.

The `/rubin/lsdb_data` filesystem is an RSP path, not a Jean-Zay path. Generate
the selected `objectId` list and images on Jean-Zay, perform the small regional
photo-z join on the RSP, then transfer the resulting Parquet table back into
the run's provenance directory.

## Dense-patch network campaign (`i < 21`, `Re > 0.6 arcsec`)

Use this download-only campaign when network access is the scarce resource and
cutout extraction will run later. It first counts the selected objects in every
DP2 patch, keeps the 6,000 densest patches, queries every selected object in
those patches, and mirrors each available complete `ugrizy` coadd once. It does
not extract cutouts or run HATS.

The exact default selection is:

```sql
i_cModelMag < 21
AND i_extendedness = 1
AND griz_model_extendedness >= 0.8
AND sersic_no_data_flag = 0
AND sersic_unknown_flag = 0
AND sersic_reff_major > 0.6
```

At the DP2 pixel scale, `Re > 0.6 arcsec` means a major-axis effective radius
larger than about 3 pixels. The fixed MMU stamp remains `160 x 160` pixels, or
about `32 x 32 arcsec`; this query does not resize or preprocess images.

The manifest contains 36,000 patch-band tasks for 6,000 patches. Each job
processes at most 18,000 tasks with eight I/O workers and one process-wide cap
of 59 request starts per minute. The downloader currently uses up to three
HTTP requests per product (SIA discovery, DataLink resolution, file transfer),
so the theoretical request floor is about 30.5 hours for the full campaign.
The observed pilot throughput implies about 36 hours, before query time,
retries, queueing, and service variation. Expected coadd storage is about
1.1 TiB. The generated selection report records the actual object count and
density before downloads start.

From a Jean-Zay login node, update the fork and restore an isolated run under
`$SCRATCH`:

```bash
export MMU_JZ_ROOT="$SCRATCH/mmu_lsst_dp2"
export MMU_REPO="$MMU_JZ_ROOT/MultimodalUniverse"
export LSST_DP2_RUN_NAME="dense_i21_reff0p6_weekend_v1"

git -C "$MMU_REPO" fetch fork feat/lsst-dp2
git -C "$MMU_REPO" switch feat/lsst-dp2
git -C "$MMU_REPO" pull --ff-only fork feat/lsst-dp2

unset LSST_DP2_ROOT LSST_DP2_HATS_OUTER
source "$MMU_REPO/scripts/lsst_dp2/jeanzay_env.sh"
read -rsp "RSP token: " RSP_TOKEN
echo
export RSP_TOKEN
mkdir -p "$LSST_DP2_ROOT/logs"
```

Submit two full task batches and a short third recovery job. Jobs are strictly
sequential, so they share neither the request budget nor the SQLite manifest
concurrently. `afterany` allows the next job to recover tasks left `running` by
a timeout; completed FITS are not downloaded again.

```bash
JOB1=$(sbatch --parsable \
  --account=jrx@cpu --partition=prepost --export=ALL \
  --output="$LSST_DP2_ROOT/logs/dense-%j.out" \
  --error="$LSST_DP2_ROOT/logs/dense-%j.err" \
  "$MMU_REPO/scripts/lsst_dp2/dense_download_prepost.slurm")

JOB2=$(sbatch --parsable --dependency="afterany:$JOB1" \
  --account=jrx@cpu --partition=prepost --export=ALL \
  --output="$LSST_DP2_ROOT/logs/dense-%j.out" \
  --error="$LSST_DP2_ROOT/logs/dense-%j.err" \
  "$MMU_REPO/scripts/lsst_dp2/dense_download_prepost.slurm")

JOB3=$(sbatch --parsable --dependency="afterany:$JOB2" \
  --account=jrx@cpu --partition=prepost --export=ALL \
  --output="$LSST_DP2_ROOT/logs/dense-%j.out" \
  --error="$LSST_DP2_ROOT/logs/dense-%j.err" \
  "$MMU_REPO/scripts/lsst_dp2/dense_download_prepost.slurm")

unset RSP_TOKEN
echo "JOB1=$JOB1 JOB2=$JOB2 JOB3=$JOB3"
```

Queue and live progress:

```bash
watch -n 30 "date; squeue -j $JOB1,$JOB2,$JOB3 \
  -o '%.18i %.2t %.12M %.30R'; du -sh '$LSST_DP2_ROOT/coadds' 2>/dev/null"
```

After a reconnect, restore the variables shown above and recover the job IDs
with `squeue -u "$USER" -n mmu-lsst-net`. Inspect one active log with
`tail -F "$LSST_DP2_ROOT/logs/dense-JOBID.out"`. The catalog selection report
is `$LSST_DP2_ROOT/catalog/objects.parquet.selection.json`.

Check manifest progress without authentication:

```bash
cd "$MMU_REPO"
"$MMU_PYTHON" -m scripts.lsst_dp2.download_coadds \
  --catalog "$LSST_DP2_ROOT/catalog/objects.parquet" \
  --mirror-root "$LSST_DP2_ROOT/coadds" \
  --manifest "$LSST_DP2_ROOT/download_manifest.sqlite" \
  --status-only
```

Patch density is a prioritization strategy for this time-bounded cache, not a
scientifically representative sampling scheme. Preserve the inventory and
selection report. A downstream training sample should explicitly assess sky,
magnitude, size, color, seeing, depth, and missing-band coverage before use.

### Extend an existing dense campaign

Increasing the number of jobs does not extend the selected footprint: jobs
reuse the existing catalog and finish once its products are resolved. To
acquire more objects, create a new run using `--extend-from`. The new run reuses
the previous global patch inventory and selects only ranks after the previous
run's last rank. `--patch-limit` is the final global rank, including previous
runs, not the number of additional patches.

For a source run covering ranks 1--6,000, a limit of 30,000 selects up to
24,000 NEW patches (ranks 6,001--30,000). The existing 6,000 patches and their
FITS stay in their original run. The extension neither redownloads them nor
retries their failures. Query parameters, inventory fingerprint, and rank
interval are recorded; incompatible selections and overlapping source output
paths are rejected. A later extension can use this new run as its source.

The following submits **10 sequential prepost jobs, each with a 20-hour
wall-time limit, one node and 8 CPUs**. Peak allocation is one node / 8 CPUs /
0 GPUs, not 80 CPUs concurrently. The allocation ceiling is 200 hours of
wall time and 1,600 CPU-hours; jobs can exit sooner. Each job attempts at most
18,000 patch-band tasks, and remaining jobs are for continuation or retries.
The new 24,000 patches require up to 144,000 products. Allow roughly 4--4.5 TiB
of additional FITS, with the actual sizes and band availability determining
the result. No extraction, HATS import or pixel validation is submitted.

On the Jean-Zay login node:

```bash
export MMU_JZ_ROOT="$SCRATCH/mmu_lsst_dp2"
export MMU_REPO="$MMU_JZ_ROOT/MultimodalUniverse"
cd "$MMU_REPO"
git pull --ff-only https://github.com/MaxRonce/MultimodalUniverse.git feat/lsst-dp2

export LSST_DP2_RUN_NAME="dense_i21_reff0p6_30k_v1"
read -rsp "RSP token: " RSP_TOKEN
echo
export RSP_TOKEN

bash scripts/lsst_dp2/submit_dense_extension.sh \
  --source-run "$MMU_JZ_ROOT/runs/dense_i21_reff0p6_weekend_v1" \
  --patch-limit 30000 \
  --jobs 10

unset RSP_TOKEN
```

The submitter sets a fresh output root, 8 download workers, and a limit of
59 request starts/minute. It uses `singleton` with the existing `mmu-lsst-net`
job name: previously submitted jobs of this user with that name must finish
before the new passes start. Jobs proceed after failures/timeouts as well as
success; a run-level file lock also rejects accidental simultaneous wrapper
launches. All related download jobs must keep this job name, and independent
manual downloaders must not run alongside them. Authentication failures cannot
be resolved by retries; use a fresh token for new submissions when needed.

A receipt with successfully submitted job IDs (no credentials) is saved to
`$MMU_JZ_ROOT/runs/dense_i21_reff0p6_30k_v1/logs/submitted-jobs.*.txt`.
If Slurm rejects a submission, the script stops and preserves earlier IDs.
If repeating the submit command, it APPENDS jobs; it does not replace or cancel
previous submissions. The on-disk inventory identifies a fixed catalogue
snapshot. If Rubin's current catalog no longer matches its counts, the query
fails rather than silently changing membership.

After submission or reconnection:

```bash
export MMU_JZ_ROOT="$SCRATCH/mmu_lsst_dp2"
export MMU_REPO="$MMU_JZ_ROOT/MultimodalUniverse"
export LSST_DP2_RUN_NAME="dense_i21_reff0p6_30k_v1"
unset LSST_DP2_ROOT LSST_DP2_HATS_OUTER MMU_PYTHON
source "$MMU_REPO/scripts/lsst_dp2/jeanzay_env.sh"
squeue -u "$USER" -n mmu-lsst-net -o "%.18i %.2t %.12M %.40R"
ls -lt "$LSST_DP2_ROOT/logs"
```

`catalog/objects.parquet.selection.json` in the NEW run contains only the
additional objects and patches. Combine totals from disjoint runs when
reporting campaign coverage. Download completion remains distinct from pixel
coverage and scientifically usable cutouts. Failed products in the original
run must still be retried there; do not change its catalog or manifest.

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

## Offline distributed cutouts from the existing mirror

The download-only campaign and this CPU workflow are independent. An extension
download can keep running while the first campaign is processed. No RSP token,
new network query, GPU, or copy of the FITS mirror is needed.

`submit_offline_cutouts.sh` submits three stages with `afterok` dependencies:

1. **Prepare:** read a consistent SQLite snapshot, retain patches with exactly
   six terminal statuses (`complete` or `unavailable`) and at least one image,
   freeze per-shard catalogs/manifests in a NEW output run. Failed/pending or
   entirely unavailable patches are excluded and counted, not relabeled.
2. **Extract + validate:** a CPU job array balanced by object counts, keeping
   each patch in exactly one shard. Each task verifies its FITS checksums,
   opens each patch once per band, writes MMU Parquet, validates every record,
   and writes a checksummed completion receipt. Ready shards are independently
   restartable. No worker loads or hashes all other workers' FITS files.
3. **Gather:** check all receipts and Parquet checksums, ingest with the shared
   MMU HATS writer and 16 local Dask workers, create the standard 10 arcsec
   margin, then check ALL final object IDs and an image record from EACH final
   partition. This stage uses one 40-core CPU allocation for memory; it is not
   a multi-node HATS import. A technical `PASS` does not certify morphology or
   photometric calibration against an independent reference.

The image contract remains `flux`, `ivar`, `mask`, `mask_bits` with shape
`(6, 160, 160)`, band order `ugrizy`, and pixel scale 0.2 arcsec (32 arcsec field).
`mask=True` means a valid pixel; rejected/non-finite/uncovered pixels have zero
inverse variance. Missing bands have `band_present=False` and zero-filled
arrays, not synthetic observations. Local PSF images remain `(6, 35, 35)` with
their validity flags. No display stretch, smoothing, resizing, or background
manipulation is applied. The source galaxy selection is preserved unchanged.

The new extractor reuses Astropy cutout slices across the image/variance/mask
planes instead of allocating a full-patch coverage map per source. Validation
decodes dense arrays directly from Arrow buffers, avoiding the former conversion
of every pixel into a Python object. Both paths have numerical parity tests.

### Submit from the Jean-Zay frontend

Use the environment already installed for downloads. Do not run `uv sync`
concurrently against that environment. All run products, caches, and job-local
temporary directories remain under `$SCRATCH`. Update the checkout before
starting this campaign; do not change the processing code or installed packages
while it is active. Code/package hashes are part of the frozen contract.

```bash
export MMU_JZ_ROOT="$SCRATCH/mmu_lsst_dp2"
export MMU_REPO="$MMU_JZ_ROOT/MultimodalUniverse"
cd "$MMU_REPO"
git pull --ff-only https://github.com/MaxRonce/MultimodalUniverse.git feat/lsst-dp2

export LSST_DP2_ACCOUNT=jrx@cpu
export LSST_DP2_CPU_PARTITION=cpu_p1

bash scripts/lsst_dp2/submit_offline_cutouts.sh \
  --source-run "$MMU_JZ_ROOT/runs/dense_i21_reff0p6_weekend_v1" \
  --run-name dense_i21_reff0p6_mmu_v1 \
  --shards 128 \
  --concurrent 16
```

128 is the number of work units, not the number of simultaneously allocated
nodes. `--concurrent 16` caps the array at 16 jobs x 8 allocated CPU cores = 128
cores; `--concurrent 32` requests a ceiling of 256 cores. Each job runs four
patch-processing threads, then single-process vectorized validation. CPU
allocation is not a measurement of utilization. Measure `TotalCPU`, `AllocCPUS`,
`Elapsed`, and `MaxRSS` with `sacct` after completion. Availability and account
quotas still control scheduling. Increase concurrency only after checking
filesystem throughput; more readers do not guarantee a proportional speedup.

The gather uses 16 processes on one CPU node, not all 40 cores for computation.
It reserves the whole node to leave memory for decoded arrays and HATS shuffle.
The default partition can be overridden with `LSST_DP2_CPU_PARTITION`; check that
your account has CPU access before submission. Each array/gather job requests
20 hours; prepare requests two hours.

**Size planning:** the earlier 50k pilot occupied approximately 53 GiB of
intermediate Parquet and 55 GiB of final HATS. Linear *storage-only* extrapolation
to 1,484,453 sources is about 1.54 TiB + 1.59 TiB, besides the existing ~1 TiB
FITS mirror. Allow several additional TiB for HATS shuffle and temporary data;
check your project quota, not just `df`. Compression and missing-band fractions
can change these estimates. Nothing is automatically deleted after success.
There is no measured wall-time guarantee for the distributed million-row run.

The snapshot count is computed when prepare runs. It will be 1,484,453 only if
the input manifest still has the same 5,999 ready patches; if the last failed
product was recovered, more rows can legitimately be included. These are
selected candidates with available images, not an automatically pure or
pixel-quality-filtered galaxy sample. The report includes band/PSF availability,
valid central 5x5 pixel counts, and present bands with no valid pixels. No
additional quality selection is silently applied.

### Monitor, reconnect, and resume

```bash
export OUT="$SCRATCH/mmu_lsst_dp2/runs/dense_i21_reff0p6_mmu_v1"
cat "$OUT"/logs/offline-jobs.*.txt
squeue -u "$USER" -n dp2-dense_i21_reff0p6_mmu_v1 \
  -o '%.22i %.2t %.12M %.40R'
ls "$OUT"/inputs/plan.json
# Substitute the array job ID from the receipt; there is one log per task.
tail -F "$OUT/logs/extract-ARRAY_ID_0.out"
find "$OUT/shards" -name validation.json | wc -l
# After gather:
cat "$OUT/validation_report.json"
```

The final catalog is
`$OUT/hats/lsst_dp2/lsst_dp2/lsst_dp2`; validated intermediate files remain in
`$OUT/shards/NNNNN/parquet`. `ingest_input/` contains only symlinks, not another
data copy. The source selection report and snapshot exclusions live under
`$OUT/inputs/`. The job receipt contains IDs, never tokens.

If a stage fails, inspect its `.err` AND `.out` logs and `sacct`. Downstream
`afterok` jobs will not run. Cancel only this campaign's pending dependent jobs
using their IDs from the receipt, wait for any running array tasks to finish,
then repeat the same submission command. Finished shards are checksummed and
reused; unfinished shards reuse their existing valid-row-count Parquet files
and undergo full record validation. A failed HATS gather is rebuilt from the
validated intermediates. Do not delete or modify source coadds or successful
receipts. A different snapshot, schema, code version, or package environment
requires a new output run name.

### Review the finished dataset locally or on W&B

First check the **gather job**, not just the extraction array: `sacct -j GATHER_ID
--format=JobID,State,ExitCode,Elapsed,MaxRSS` must show `COMPLETED` and `0:0`.
The final `validation_report.json` must exist with `status=PASS`.

`review_dataset.py` checks that this report matches the frozen plan, compares
its counts with **all actual main-catalog Parquet footers**, and validates a
reproducible uniform random sample of image records from the final HATS. It
does not rerun the full pixel/checksum audit or modify any source files. The
original pipeline validated every intermediate record and all final object IDs;
the small review is an additional readback, not a scientific certification.
The sample has no extra S/N or quality cut. It represents this density-selected
dataset, not the whole DP2 sky. Unavailable bands and invalid PSFs are retained
with their flags, never replaced by simulated measurements.

Run the export on a CPU node; no network access or RSP token is needed. Updating
only the review utilities does not change the frozen processing contract. Do
not update dependencies or processing code while production jobs are active.

```bash
export MMU_JZ_ROOT="$SCRATCH/mmu_lsst_dp2"
export MMU_REPO="$MMU_JZ_ROOT/MultimodalUniverse"
export LSST_DP2_RUN_NAME=dense_i21_reff0p6_mmu_v1
unset LSST_DP2_ROOT LSST_DP2_HATS_OUTER
source "$MMU_REPO/scripts/lsst_dp2/jeanzay_env.sh"
cd "$MMU_REPO"
git pull --ff-only https://github.com/MaxRonce/MultimodalUniverse.git feat/lsst-dp2

srun --account=jrx@cpu --partition=cpu_p1 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --hint=nomultithread --time=01:00:00 \
  "$MMU_PYTHON" -u -m scripts.lsst_dp2.review_dataset export \
  --run-root "$LSST_DP2_ROOT" --output-dir "$LSST_DP2_ROOT/review24" \
  --count 24 --stretch linear
```

An existing output directory is refused to avoid mixing reviews. Use another
name, for example `review24_asinh`, for `--stretch asinh`. The export includes:

- `review.json`: counts, validation scope, seed, object metadata, band/PSF
  availability, original pipeline report, and rendering parameters;
- one PNG per object: six columns `ugrizy`, three rows **flux / ivar / bad
  pixels**, always the full `160x160` stamp, no crop or smoothing;
- one NPZ per object: unchanged arrays, units, mask bits, PSF kernels, and
  provenance, readable with `np.load(path, allow_pickle=False)`;
- `index.html`: a local gallery with links to the raw NPZs.

Only the display is scaled: flux uses the 0.5th/99.5th percentiles across valid
pixels in all bands of one object; ivar uses zero to the 99.5th percentile.
Both ranges are shared across that object's bands, **not across objects**.
White mask pixels are bad (`image.mask=False`); a black mask panel means valid
pixels, not missing mask information. An unavailable band has a white bad-pixel
panel and zero flux/ivar. Display percentiles can hide faint structures or clip
bright cores; the NPZ values are unaffected.

From your **local machine**, retrieve only the small review, not the full HATS:

```bash
export REVIEW_LOCAL=/home/maxime/src/LSST_cutouts/reviews/dense_i21_mmu
mkdir -p "$REVIEW_LOCAL"
rsync -avh --progress -e "ssh -J mr287471@hubble.extra.cea.fr" \
  urx63nr@jean-zay.idris.fr:/lustre/fsn1/projects/rech/jrx/urx63nr/mmu_lsst_dp2/runs/dense_i21_reff0p6_mmu_v1/review24/ \
  "$REVIEW_LOCAL/"
xdg-open "$REVIEW_LOCAL/index.html"
```

Optional W&B publication uses an isolated `uv run` environment, **not** an
installation into the frozen MMU environment. Run locally after updating the
local checkout too. Authenticate through `wandb login` if needed; do not put
API keys in shell history. Use a private project restricted to collaborators
authorized to access DP2 pixel data; the PNGs are still pixel-level products.
Nothing is uploaded by the export command. The explicit publisher uploads only
PNGs and summary metadata, not the raw NPZ files.

```bash
cd /home/maxime/src/MMU/MultimodalUniverse
git pull --ff-only https://github.com/MaxRonce/MultimodalUniverse.git feat/lsst-dp2
uv run --no-project --with wandb --with pillow --with numpy \
  python scripts/lsst_dp2/publish_review.py \
  --review-dir "$REVIEW_LOCAL" \
  --entity maxronce-universit-de-tours --project lsst-dp2-mmu
```

The command prints the run URL. Open the `cutouts` Table and its `panels`
column to inspect each object. `--mode offline` writes a local W&B run without
uploading. See the [W&B Tables documentation](https://docs.wandb.ai/guides/track/log/log-tables).

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
