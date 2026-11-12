# JADES DR5 → MMU v2: Rusty operator runbook

This is the **JWST/NIRCam + MIRI** pipeline. It does **not** create the separate
`jades_hst` dataset, because the HST mosaics have not been downloaded yet.
All cutouts are **160 × 160 native pixels**, not reprojected to a shared scale.

## 0. Resume your Rusty environment

```bash
cd /mnt/ceph/users/polymathic/mronceray/MultimodalUniverse-jades
git pull --ff-only origin feat/jades-dr5
export JADES_RAW=/mnt/ceph/users/polymathic/external_data/Astro/JADES_DR5
export JADES_RUN=/mnt/ceph/users/polymathic/mronceray/jades_dr5_runs
mkdir -p "$JADES_RUN"
rg -n 'jades-v2-multimodal-2' scripts/jades/build_parent_sample_hats.py
```

**Do not start the v2 run unless the final command finds the schema version.**
If it prints nothing, the new implementation has not yet been synced to your
Rusty checkout; fetching the previous remote branch is not enough.

## 0a. Reserve a Rusty CPU worker inside tmux (Slurm)

Do not run the image extraction on a `rusty` login node. If one is already
running there, interrupt it with `Ctrl+C`; its existing committed Parquet
checkpoints remain reusable with the **same** command and scratch directory.

Start or reattach a tmux session on the login node:

```bash
tmux new -s jades-build
# If it already exists, instead use: tmux attach -t jades-build
```

**Inside tmux**, request an interactive compute shell through Slurm:

```bash
command -v srun || module load slurm
sinfo -p genx -o '%P %a %l %D %C'
srun --job-name=jades-dr5 --partition=genx \
  --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=32G --time=04:00:00 --pty bash -l
```

Wait for the allocation, then verify that `hostname` is a **compute** node,
not `rusty1`–`rusty4`, and that `SLURM_JOB_ID` is set:

```bash
hostname
echo "$SLURM_JOB_ID"
cd /mnt/ceph/users/polymathic/mronceray/MultimodalUniverse-jades
export JADES_RAW=/mnt/ceph/users/polymathic/external_data/Astro/JADES_DR5
export JADES_RUN=/mnt/ceph/users/polymathic/mronceray/jades_dr5_runs
command -v uv
```

If `srun` rejects `genx`, account, memory or wall time, inspect the actual
Slurm error and available partitions before changing resource flags. The
interactive allocation lasts at most four hours; resume the identical
checkpointed command in a new allocation if the limit expires. **No GPU** is
needed. The image extraction is currently mostly serial, so reserving four
CPUs does not itself create four Python image workers; moving from login to a
compute node is the first goal. Profile real I/O before attempting parallelism.

Detach with `Ctrl+B`, then `D`; reattach from a new login shell with
`tmux attach -t jades-build`. To release the allocation deliberately, run
`exit` from the compute shell.

If Rusty `uv` reports that the project SCM version cannot be determined, use
`SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run ...` with the same command.

## 1. Verify input inventory

```bash
find "$JADES_RAW/mosaics/GOODS-S" "$JADES_RAW/mosaics/GOODS-N" -maxdepth 1 -name '*.fits' | wc -l
find "$JADES_RAW/mosaics/MIRI" -maxdepth 1 -name '*.fits' | wc -l
ls -lh "$JADES_RAW/catalogs/GOODS-S.fits" "$JADES_RAW/catalogs/GOODS-N.fits"
```

Expected: **34 NIRCam + 18 MIRI mosaics**, and both DR5 photometric catalogs.
The downloader size-checked NIRCam; the MIRI GitHub downloads were SHA-256
verified by the operator. A FITS header check does not validate the entire
1.1-TB NIRCam payload cryptographically.

## 2. Generate and validate the pinned manifest

```bash
uv run python -m scripts.jades.make_manifest \
  --raw-root "$JADES_RAW" \
  --output "$JADES_RUN/jades_dr5_manifest.json"
```

This inspects FITS headers without reading entire mosaic arrays. It fails if
there are missing/extra products, SCI/ERR geometry mismatches, unexpected
units, or missing SCI pixel area/WCS. The manifest pins 52 mosaics and two
catalogs. The selected products are native DR5 mosaics, not PSF-homogenized
images; the `ERR` map is a resampled per-pixel sigma, **not** a fully
correlated-noise-aware aperture uncertainty.

## 3. Run a tiny Parquet-only smoke test (new v2 schema)

The limit is **per GOODS field** (100 for GOODS-S and 100 for GOODS-N).
Do not reuse this scratch directory for a larger/different selection. Previous
`scratch_smoke100` and `scratch_miri100` belong to the old schema: **leave
them intact** and use the new `_v2` directories below.

```bash
uv run python -m scripts.jades.build_parent_sample_hats \
  --instrument jwst \
  --manifest "$JADES_RUN/jades_dr5_manifest.json" \
  --raw-root "$JADES_RAW" \
  --scratch-dir "$JADES_RUN/scratch_smoke100_v2" \
  --image-size 160 --max-objects 100 \
  --chunk-size 4 --rows-per-file 32 --skip-hats
```

To check **MIRI coverage specifically**, use a **different scratch**:

```bash
uv run python -m scripts.jades.build_parent_sample_hats \
  --instrument jwst \
  --manifest "$JADES_RUN/jades_dr5_manifest.json" \
  --raw-root "$JADES_RAW" \
  --scratch-dir "$JADES_RUN/scratch_miri100_v2" \
  --image-size 160 --sample-mode miri --max-objects 100 \
  --chunk-size 4 --rows-per-file 32 --skip-hats
```

`--sample-mode miri` selects the first 100 catalog sources in each field with
a positive MIRI exposure flag. Actual mosaic coverage and valid ERR/SCI pixels
are still verified; there is **no guarantee** that every selected source has
all three MIRI filters.

## 4. Audit saved images and make preview PNGs

```bash
uv run --extra viz python -m scripts.jades.inspect_cutouts \
  --scratch-dir "$JADES_RUN/scratch_smoke100_v2" \
  --output "$JADES_RUN/smoke100_v2_audit.json" \
  --max-samples 24 \
  --require-photoz \
  --preview-dir "$JADES_RUN/smoke100_v2_previews"
```

For the MIRI-focused sample:

```bash
uv run --extra viz python -m scripts.jades.inspect_cutouts \
  --scratch-dir "$JADES_RUN/scratch_miri100_v2" \
  --output "$JADES_RUN/miri100_v2_audit.json" \
  --max-samples 24 \
  --require-photoz \
  --require-any-band f770w f1280w f1500w \
  --preview-dir "$JADES_RUN/miri100_v2_previews"
```

The audit checks 2-D shapes, non-finite pixels, positive IVAR only where the
mask is true, unique lowercase band names, PSF/pixel-scale alignment, and
photo-z PDF bin consistency. It requires exactly **160 × 160** pixels per
band, samples up to **24 distinct objects across the full Parquet shard
sequence** (not just one per file), and rejects an audit with no native or
Kron photo-z estimates. It reports photo-z/photometry coverage, MIRI/NIRCam
joint-source counts, photometry availability per measurement method
(CIRC3/KRON/KRON_CONV), and per-band valid-pixel fractions. Preview RGB is
display-scaled and not a scientific flux measurement.

The final image struct is `image.{band,flux,ivar,mask,psf_fwhm,scale}`.
NIRCam and MIRI share this one struct in a **single** `jades_jwst` catalog.
Top-level `phot_f090w_circ3_flux_njy`, `phot_f770w_kron_flux_njy`, etc.,
come from catalog HDUs whether or not that filter has a matching image.
`photoz` and `photoz_kron` are two **photometric** EAZY estimates; `z_spec`
is separate and optional. Image `psf_fwhm` is **null** unless a measured
`psf_fwhm_arcsec` is supplied per mosaic in the manifest (arcseconds).

**Check the audit and inspect the PNGs before any large run.** Inspect that
the galaxy is centered, no cutout is all zeros, and MIRI bands genuinely show
coverage rather than just flag values.

## 5. Make a small HATS test

Repeat exactly the smoke command, keeping the same scratch parameters, but
**omit `--skip-hats`** and add `--output-root` plus staging:

```bash
uv run python -m scripts.jades.build_parent_sample_hats \
  --instrument jwst \
  --manifest "$JADES_RUN/jades_dr5_manifest.json" \
  --raw-root "$JADES_RAW" \
  --scratch-dir "$JADES_RUN/scratch_smoke100_v2" \
  --output-root "$JADES_RUN/hats_smoke100_v2" \
  --tmp-dir "$JADES_RUN/hats_tmp_v2" \
  --image-size 160 --max-objects 100 \
  --chunk-size 4 --rows-per-file 32 \
  --pixel-threshold 256 --hats-batch-size 4 --ingest-workers 2
```

The extractor will detect the completed field checkpoints and skip all
previously saved cutouts. Only the HATS ingestion stage is rerun.

## 6. Review catalog magnitude statistics

```bash
uv run python -m scripts.jades.audit_catalog \
  "$JADES_RAW/catalogs/GOODS-S.fits" \
  "$JADES_RAW/catalogs/GOODS-N.fits" \
  --bands F090W F115W F150W F200W F277W F356W F410M F444W \
  --mag-band F277W --mag-cuts 25 26 27 28 29 30 \
  --output "$JADES_RUN/catalog_snr_audit.json"
```

These are diagnostic counts, **not a measurement of survey completeness**.
Select any production `--mag-max` from science/completeness analysis, not from
this histogram alone. No selection is imposed unless `--mag-band` and
`--mag-max` are supplied together.

## 7. Scale gradually; estimate storage from a pilot

```bash
du -sh "$JADES_RUN/scratch_smoke100_v2" "$JADES_RUN/hats_smoke100_v2"
df -h "$JADES_RUN"
```

The full sample has **485,510 source catalog rows before image availability
and optional magnitude cuts**. Storing 160×160 `flux`, `ivar` and `mask` arrays
for many bands per object can require **terabytes of output and staging**;
HATS margin files can increase size further. Check your actual Ceph quota.
Do not assume 1.1 TB of raw images implies a 1.1 TB final dataset.

Pilot 1,000 objects per field, using **new** scratch/output paths:

```bash
uv run python -m scripts.jades.build_parent_sample_hats \
  --instrument jwst \
  --manifest "$JADES_RUN/jades_dr5_manifest.json" \
  --raw-root "$JADES_RAW" \
  --scratch-dir "$JADES_RUN/scratch_pilot1000_v2" \
  --image-size 160 --max-objects 1000 \
  --chunk-size 4 --rows-per-file 64 --skip-hats
```

After inspecting disk usage, CPU, memory, output quality and catalog cuts,
start a full extraction with `--max-objects` **omitted**, in a dedicated
`scratch_production` directory. For production, run the same command again
without `--skip-hats`, using a distinct `--output-root` and enough staging
space. Do not launch full HATS ingestion until the pilot is validated.

### Crash recovery

Running the **identical** extraction command against the same scratch root
resumes from the latest committed per-field Parquet checkpoint. A crashed
in-progress shard is reconstructed; completed shards are not re-extracted.
If you change magnitude cuts, sample mode, image size, catalog or manifest,
create a **new** scratch directory. Reusing a scratch with different settings
is an error by design. The checkpoint also records the schema version: old
JADES Parquet shards cannot be reused after a schema change. Do not manually
edit `.progress.json` or `.done` files.

### What is intentionally not validated by these scripts

- The true end-to-end photometric calibration/PSF model for every filter;
- The science-level treatment of *correlated* noise in mosaic pixels;
- Full-depth completeness, morphology or photo-z selection;
- Full build runtime and final dataset size on Rusty (must be measured);
- HST mosaic production (a separate `jades_hst` workflow is pending).

For MIRI, the `MIRI` HDU supplies CIRC3 and native KRON photometry rather
than NIRCam's `CIRC`/`KRON` HDUs. `PHOTOZ` and `PHOTOZ_KRON` supply separate
EAZY estimates (`z_a`, `z_peak`, `z_ml`, confidence intervals and 22-bin
redshift distributions); spectroscopic `z_spec` is independent. The selected
input mosaic is preserved in `image_sources`, aligned to lowercase
`image.band`. A source with no MIRI cutout may still have MIRI photometry.

## 8. Distributed JWST production (Slurm array + multi-node Dask)

**This is the production path for the tutor-approved specification.** One
galaxy row contains its available native-pixel NIRCam and MIRI bands. Each
band has real pixel IVAR, mask, pixel scale and nullable PSF FWHM. All catalog
photometry, both EAZY photo-z estimates and separate `z_spec` are retained.
No magnitude or SNR cut is applied. A filter is included only when the galaxy
centroid has a valid SCI/ERR pixel in that filter; boundary pixels elsewhere
in a stamp do not create phantom bands.

An extraction array performs **independent contiguous slices** of the
GOODS-S and GOODS-N catalogs. Each Slurm task writes to
`$JADES_SCRATCH/shard_XXXX_of_NNNN/`, with separate `.progress.json`, `.done`
and Parquet checkpoints. An incomplete task can be resubmitted without
regenerating completed tasks. The follow-up HATS ingestion validates **every**
shard, checks Parquet row counts, and uses Dask workers on multiple nodes.
The resulting dataset is still **one `jades_jwst` HATS collection**.

### 8.1 Prepare on the Rusty login node (not inside a compute allocation)

The code must first be **committed and pushed to the Git branch**, then pulled
on Rusty. Uncommitted Codex workspace edits cannot be fetched by `git pull`.

```bash
cd /mnt/ceph/users/polymathic/mronceray/MultimodalUniverse-jades
git pull --ff-only origin feat/jades-dr5
rg -n -- '--num-shards|--ingest-only' scripts/jades/build_parent_sample_hats.py

export JADES_RAW=/mnt/ceph/users/polymathic/external_data/Astro/JADES_DR5
export JADES_RUN=/mnt/ceph/users/polymathic/mronceray/jades_dr5_runs
export JADES_MANIFEST="$JADES_RUN/jades_dr5_manifest.json"
mkdir -p "$JADES_RUN/logs"

sinfo -p genx -o '%P %a %l %D %C'
```

Run `uv run --extra dev pytest -q scripts/jades/test_jades.py` before
submitting jobs. The tests use synthetic data and are inexpensive.

### 8.2 Distributed pilot: 1,000 sources per GOODS field

Use **8 total shards, 4 concurrent Slurm workers**, to measure actual Ceph
throughput and memory before submitting the entire catalog:

```bash
export JADES_NUM_SHARDS=8
export JADES_MAX_OBJECTS=1000
export JADES_SCRATCH="$JADES_RUN/scratch_jwst_pilot1000_dist_v2"

PILOT_JOB=$(sbatch --parsable --export=ALL \
  --job-name=jades-pilot --partition=genx \
  --array=0-7%4 --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --mem=16G --time=04:00:00 \
  --output="$JADES_RUN/logs/pilot_%A_%a.out" \
  scripts/jades/slurm_extract.sh)
echo "$PILOT_JOB"
```

Monitor `squeue -j "$PILOT_JOB"` and inspect
`$JADES_RUN/logs/pilot_<job>_<task>.out`. Once all array elements have
**COMPLETED successfully**, audit the *whole directory*, including its
per-task subdirectories:

```bash
uv run --extra viz python -m scripts.jades.inspect_cutouts \
  --scratch-dir "$JADES_SCRATCH" \
  --output "$JADES_RUN/pilot1000_dist_v2_audit.json" \
  --max-samples 48 --require-photoz \
  --preview-dir "$JADES_RUN/pilot1000_dist_v2_previews"

du -sh "$JADES_SCRATCH"
df -h "$JADES_SCRATCH"
```

Before full production, test HATS import for this pilot on an allocated
worker (e.g. `srun --partition=genx --cpus-per-task=4 --mem=32G
--time=04:00:00 --pty bash -l`; re-export JADES variables inside that shell):

```bash
uv run python -m scripts.jades.build_parent_sample_hats \
  --instrument jwst --manifest "$JADES_MANIFEST" --raw-root "$JADES_RAW" \
  --scratch-dir "$JADES_SCRATCH" --num-shards 8 --ingest-only \
  --output-root "$JADES_RUN/hats_pilot1000_dist_v2" \
  --tmp-dir "$JADES_RUN/hats_pilot_tmp_v2" \
  --hats-batch-size 4 --ingest-workers 4 --pixel-threshold 256
```

Do not run an identical HATS command again against an output path with existing
files. Use a **new** `--output-root` for a retry. Original outputs are never
removed by this JADES CLI, even if an old HATS import failed.

### 8.3 Full JADES extraction: no cuts

Return to the **login node**. Choose a fresh scratch root; **remove the
pilot-only environment limit with `unset`**. The following uses 32 total
shards and up to 8 simultaneous workers. Change `%8` only if your Slurm quota
and measured shared-filesystem throughput support more concurrent requests.
Each task stays restartable when its allocation expires.

```bash
unset JADES_MAX_OBJECTS
export JADES_NUM_SHARDS=32
export JADES_SCRATCH="$JADES_RUN/scratch_jades_jwst_full_dist_v2"

EXTRACT_JOB=$(sbatch --parsable --export=ALL \
  --job-name=jades-extract --partition=genx \
  --array=0-31%8 --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --mem=16G --time=12:00:00 \
  --output="$JADES_RUN/logs/extract_%A_%a.out" \
  scripts/jades/slurm_extract.sh)
echo "$EXTRACT_JOB"
```

Monitoring:

```bash
squeue -j "$EXTRACT_JOB"
sacct -j "$EXTRACT_JOB" --format=JobID,State,Elapsed,MaxRSS
du -sh "$JADES_SCRATCH"
```

If only array tasks 5 and 17 failed, resubmit those indices with **the same
total `JADES_NUM_SHARDS=32`, same scratch directory and extraction settings**:

```bash
sbatch --export=ALL --job-name=jades-retry --partition=genx \
  --array=5,17%2 --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --mem=16G --time=12:00:00 \
  --output="$JADES_RUN/logs/retry_%A_%a.out" \
  scripts/jades/slurm_extract.sh
```

These task indices are **examples**, not tasks to rerun automatically. Check
`sacct` and logs before deciding which ones actually failed.

### 8.4 Distributed HATS ingest across several compute nodes

Start this stage **only after** every extraction task is complete, after
reviewing the image audit and confirming available Ceph space. HATS ingestion
can require substantial additional temporary space for partitioning and margin
creation. It is a separate Dask stage, not 32 independent HATS catalogs.

Choose a **new** output and scheduler filename for each attempt, and keep the
staging directory on shared Ceph so every worker can access it:

```bash
export JADES_HATS_OUT="$JADES_RUN/hats_jades_jwst_full_dist_v2"
export JADES_HATS_TMP="$JADES_RUN/hats_tmp_jwst_full_dist_v2"
export JADES_DASK_FILE="$JADES_RUN/dask_jwst_full_dist_v2.json"
export JADES_DASK_WORKERS=8
export JADES_DASK_WORKER_MEMORY=12GiB

DASK_SCHED_JOB=$(sbatch --parsable --export=ALL \
  --job-name=jades-scheduler --partition=genx \
  --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --mem=4G --time=24:00:00 \
  --output="$JADES_RUN/logs/dask_scheduler_%j.out" \
  scripts/jades/slurm_dask_scheduler.sh)
echo "$DASK_SCHED_JOB"
```

When that job is RUNNING and the scheduler file exists:

```bash
DASK_WORKER_JOB=$(sbatch --parsable --export=ALL \
  --job-name=jades-dask-worker --partition=genx \
  --array=0-7%8 --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --mem=16G --time=24:00:00 \
  --output="$JADES_RUN/logs/dask_worker_%A_%a.out" \
  scripts/jades/slurm_dask_worker.sh)
echo "$DASK_WORKER_JOB"
```

Then submit the HATS client job. It will **wait for all 8 Dask workers**
before beginning the import; it verifies the 32 completed extraction shards
and their actual Parquet row counts first:

```bash
HATS_JOB=$(sbatch --parsable --export=ALL \
  --job-name=jades-hats --partition=genx \
  --nodes=1 --ntasks=1 --cpus-per-task=2 \
  --mem=16G --time=24:00:00 \
  --output="$JADES_RUN/logs/hats_%j.out" \
  scripts/jades/slurm_hats_ingest.sh)
echo "$HATS_JOB"
```

Wait until `sacct -j "$HATS_JOB"` reports **COMPLETED**, then inspect:

```bash
du -sh "$JADES_HATS_OUT"
find "$JADES_HATS_OUT" -maxdepth 3 -type f -name 'properties'
```

The collection is named `jades_jwst`. The HATS inner catalog is typically
`$JADES_HATS_OUT/jades_jwst/jades_jwst` (with a margin catalog alongside it).
After verifying successful import, explicitly release the two long-running
Dask allocations with `scancel "$DASK_WORKER_JOB" "$DASK_SCHED_JOB"`.
The scheduler file may already disappear when Dask stops; always select a new
name on the next attempt.

**Important constraints:** Slurm partitions/accounts/wall-time limits are
site-specific; adjust these resource requests if Rusty rejects them. Scale
the number of *concurrent* workers gradually because Ceph I/O can saturate.
An interrupted extraction is resumable at committed Parquet checkpoints;
an interrupted HATS import currently requires a fresh HATS output path.
