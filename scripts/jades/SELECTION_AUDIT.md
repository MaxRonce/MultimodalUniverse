# JADES magnitude / SNR selection audit

Run this **independent, read-only job during extraction**. It reads scalar
photometry from the two FITS catalogs, without opening the image mosaics or
modifying any dataset files. It finishes after one pass; repeated polling is
unnecessary for the complete parent-catalog statistics.

The Slurm launcher uses `uv run --no-project --isolated` with only numpy,
astropy, matplotlib and pyarrow. It does not build/install MMU or modify the
repository's `.venv`. This avoids failures when setuptools-scm encounters
non-version tags such as `demo-2026-04-07-hackathon`, and ignores any active
AION environment inherited by the job. First launch may install the small audit
dependency set into uv's cache.

From the MMU repository root on the Rusty login node:

```bash
git switch feat/jades-dr5
git pull --ff-only origin feat/jades-dr5

export JADES_RAW=/mnt/ceph/users/polymathic/external_data/Astro/JADES_DR5
export JADES_RUN=/mnt/ceph/users/polymathic/mronceray/jades_dr5_runs
export JADES_MANIFEST="$JADES_RUN/jades_dr5_manifest.json"
export JADES_AUDIT_MAG_BAND=F277W
export JADES_AUDIT_OUT="$JADES_RUN/selection_audit_f277w"
mkdir -p "$JADES_RUN/logs"

AUDIT_JOB=$(sbatch --parsable --export=ALL \
  --job-name=jades-audit --partition=genx \
  --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --mem=8G --time=02:00:00 \
  --output="$JADES_RUN/logs/selection_audit_%j.out" \
  scripts/jades/slurm_audit_selection.sh)
echo "$AUDIT_JOB"
```

The job has no dependency on the extraction array and does not consume one of
its eight array slots. Slurm still decides when resources are available.

Read results with:

```bash
sacct -j "$AUDIT_JOB" --format=JobID,State,Elapsed,ExitCode
cat "$JADES_AUDIT_OUT/report.txt"
ls -lh "$JADES_AUDIT_OUT"
```

## Definitions

- Reference magnitude: **F277W native KRON AB**, calculated as
  `31.4 - 2.5 log10(flux_nJy)` for positive, finite flux. Change
  `JADES_AUDIT_MAG_BAND` and `JADES_AUDIT_OUT` to study F356W or F444W.
  These yield different selections; there is no single band-independent magnitude.
- Cumulative cuts: `m < 21, 22, ..., 29` (strict upper bound).
- Disjoint magnitude intervals: `<21`, `[21,22)`, ..., `[28,29)`, `>=29`.
  Missing magnitudes appear in the `all` row separately.
- SNR: **signed catalog CIRC3 flux / CIRC3_e**, for finite flux and positive,
  finite uncertainty. It is an aperture measurement, not a pixel SNR or a
  quadrature sum over cutout pixels. MIRI columns come from the `MIRI` HDU.
- SNR bins: `<0`, `[0,2)`, `[2,3)`, `[3,5)`, `[5,10)`, `[10,20)`,
  `[20,50)`, `[50,100)`, `>=100`, plus **missing**.
  Negative and zero flux are valid SNR measurements. They do not produce a magnitude.
- The per-band histograms report both percentage of **all selected sources**
  and percentage of **selected sources with valid SNR in that band**.
  Missing values are never silently dropped from the first denominator.
- Multiband tables count sources with SNR `>=2, 3, 5, 10` in at least `1, 2, 4`
  bands, separately for JWST, NIRCam and MIRI. They also include the maximum
  SNR over each instrument's bands as explicitly named histogram metrics.
- Results include GOODS-S, GOODS-N and their combined `ALL` counts. Identical
  numeric source IDs in different fields remain separate sources.

These are **source counts**, without a star/quality cut. All JWST bands with
catalog flux/error columns are discovered automatically; photometric availability
does not establish image coverage. No magnitude or SNR cut is applied to the build.

## Outputs

| File | Contents |
|---|---|
| `counts.csv` | Counts and fractions by field, cumulative magnitude cut and magnitude interval |
| `snr_bins.csv` | SNR counts and both percentage denominators for every magnitude selection and band |
| `detections.csv` | Joint magnitude/SNR yields requiring at least 1, 2 or 4 bands |
| `summary.json` | Definitions, input metadata, assumptions and count tables |
| `report.txt` | Small terminal-readable cumulative-count table |
| `selection_catalog.png` / `.pdf` | Scientific overview figure for the complete parent catalogs |

## Optional: sources already extracted

To report the **actual checkpointed subset as well as the complete parent
catalog**, set the following before submitting the same Slurm job:

```bash
export JADES_AUDIT_SCRATCH="$JADES_RUN/scratch_jades_jwst_full_dist_v2"
export JADES_AUDIT_OUT="$JADES_RUN/selection_audit_f277w_snapshot"
```

The audit reads only `field` and `object_id` from committed Parquet files.
It excludes `.part` files and files not yet recorded in the extraction checkpoint.
It checks for unknown and duplicate IDs, without reading cutout cubes.
Rows tagged `population=extracted` and `selection_extracted.png/.pdf` describe
this snapshot. Their SNR values still come from catalog photometry, independently
of which image bands are present in an individual cutout.

**During extraction this subset can be spatially biased; its fractions must not
be extrapolated to the final dataset.** Re-run after all extraction shards finish
to obtain the retained-dataset selection yields. The audit does not validate
cutout pixel quality; use `inspect_cutouts` for that.

For parent-catalog-only statistics again, use `unset JADES_AUDIT_SCRATCH`.

## Direct CLI / tests

```bash
uv run --no-project --isolated --python 3.11 \
  --with numpy --with astropy --with matplotlib --with pyarrow \
  python -m scripts.jades.audit_selection \
  --raw-root "$JADES_RAW" --manifest "$JADES_MANIFEST" \
  --mag-band F277W --mag-cuts 21 22 23 24 25 26 27 28 29 \
  --output-dir "$JADES_AUDIT_OUT" --plots

uv run --no-project --isolated --python 3.11 \
  --with numpy --with astropy --with matplotlib --with pyarrow --with pytest \
  python -m pytest -q scripts/jades/test_audit_selection.py
```

`--bands`, `--aperture` and `--snr-edges` are configurable. Omit `--plots` to
avoid the optional matplotlib dependency. FITS `TUNIT` values are converted to
nJy; absent units use the JADES nJy convention and are recorded explicitly in
`summary.json`. Incompatible units and misaligned catalog IDs fail the audit.
