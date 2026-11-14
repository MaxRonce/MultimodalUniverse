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

### 8.3a Fast, checkpoint-compatible recovery of extraction array 7453873

The first full run used `--array=0-31%8`, one CPU and 16 GiB per task. Its
checkpoint/Parquet layout **must be retained**: the 32 shards contain separate
`GOODS-S` and `GOODS-N` `.progress.json`, `.done` and Parquet files. Low CPU
utilization and highly variable runtime suggest Ceph page faults / I/O waits,
but only a representative Rusty benchmark can establish the speedup.

The accelerated extractor makes two *execution-only* changes:

1. `JADES_SPLIT_FIELDS=1` schedules **64 independent Slurm tasks**: indices
   `0..31` extract GOODS-S shards `0..31`; indices `32..63` extract GOODS-N
   shards `0..31`. Each task writes to the existing field-specific filenames
   under **the same 32 shard directories**. Completed `.done` fields return
   immediately without reading mosaics.
2. `JADES_EXTRACT_THREADS=4` overlaps blocking FITS cutout reads in each task;
   `JADES_SPATIAL_BATCH_SIZE=256` groups nearby pixel coordinates within a
   *bounded* batch. Results are serialized back in the exact original catalog
   order, so old `next_index`, file numbers and fingerprints remain valid.

**Migration on the login node.** First record the current state, then cancel
**only** the old extraction array. The last committed Parquet file per field
and its `.progress.json` are preserved; at most its unfinished temporary file
and records since the last checkpoint will need to be recomputed. This does
**not** delete the ~600 GiB of saved outputs.

```bash
export JADES_RAW=/mnt/ceph/users/polymathic/external_data/Astro/JADES_DR5
export JADES_RUN=/mnt/ceph/users/polymathic/mronceray/jades_dr5_runs
export JADES_MANIFEST="$JADES_RUN/jades_dr5_manifest.json"
export JADES_SCRATCH="$JADES_RUN/scratch_jades_jwst_full_dist_v2"
export JADES_NUM_SHARDS=32
unset JADES_MAX_OBJECTS

sacct -X -j 7453873 --format=JobID,State,Elapsed,ExitCode
find "$JADES_SCRATCH" -name '*.done' | wc -l
du -sh "$JADES_SCRATCH"

scancel 7453873
squeue -r -j 7453873
```

Wait until the final `squeue` prints **no active tasks** (including
`COMPLETING`). Never run old and new jobs against the same scratch at once:
the old code does not acquire the new field lock files.

Update the Rusty repository *after the old tasks stop*:

```bash
cd /mnt/ceph/users/polymathic/mronceray/MultimodalUniverse-jades
git pull --ff-only origin feat/jades-dr5
uv run --extra dev python -m pytest -q scripts/jades/test_jades.py

export JADES_SPLIT_FIELDS=1
export JADES_EXTRACT_THREADS=4
export JADES_SPATIAL_BATCH_SIZE=256

# Verify the mapping without touching any images or checkpoints.
SLURM_ARRAY_TASK_ID=15 SLURM_SUBMIT_DIR="$PWD" JADES_PLAN_ONLY=1 \
  bash scripts/jades/slurm_extract.sh  # GOODS-S, shard 15
SLURM_ARRAY_TASK_ID=47 SLURM_SUBMIT_DIR="$PWD" JADES_PLAN_ONLY=1 \
  bash scripts/jades/slurm_extract.sh  # GOODS-N, shard 15
```

**Start with 24 concurrent jobs**, each assigned four CPUs and 32 GiB. This
is already up to 96 allocated CPUs, with up to 64 independently restartable
tasks. Slurm chooses the compute nodes; the script does not run on `rusty3`.
More jobs is not automatically faster on a shared Ceph filesystem.

```bash
FAST_JOB=$(sbatch --parsable --export=ALL \
  --job-name=jades-fast --partition=genx \
  --array=0-63%24 --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=32G --time=2-00:00:00 \
  --output="$JADES_RUN/logs/fast_%A_%a.out" \
  scripts/jades/slurm_extract.sh)
echo "FAST_JOB=$FAST_JOB"
```

The first checkpoint log now reports `scanned/s`. Compare fresh logs from
`GOODS-N` unfinished shards with the previous run (shard 15 previously reached
only 2688/5661 GOODS-N cutouts before a 12-hour timeout):

```bash
squeue -r -j "$FAST_JOB"
tail -n 15 "$JADES_RUN/logs/fast_${FAST_JOB}_47.out"
find "$JADES_SCRATCH" -name '*.done' | wc -l  # 64 field markers in total
```

After measuring throughput and observing healthy memory/IO performance, raise
the array concurrency **without restarting it**:

```bash
scontrol update JobId="$FAST_JOB" ArrayTaskThrottle=32
```

If the throughput degrades at 32 simultaneous tasks, lower it to 16 or 24.
Increasing concurrency indefinitely can make all tasks slower, especially if
the shared filesystem is saturated. The 48-hour walltime is a fail-safe, **not
a prediction that extraction will take 48 hours**. Finishing in one day is a
goal to measure, not something this script can guarantee without Rusty I/O
benchmarks.

At completion, use the original HATS verification/ingestion described in
section 8.4 with `--num-shards 32`, **not 64**. The two fields have been split
across 64 *jobs*, not into 64 new output shard directories. Check the array's
`sacct` states and all 64 `.done` files before starting ingestion. If a task
fails, resubmit its exact **new** array index with the same environment and
scratch; its field-specific checkpoint is still reusable. The new extractor
uses a nonblocking per-field lock so accidental duplicate fast jobs refuse to
overwrite active output.

### 8.4 Distributed HATS ingest across several compute nodes

Run `7462727` still reported no histograms with the sparse mapper. Its scheduler
log reports an unresponsive event loop for 62.61 seconds during mapping, followed
by client reconnects. The subsequent scheduler SIGTERM is consistent with the
submitted cleanup job, not evidence that Slurm caused the initial mapping failure.
No individual worker computation exception appears in the supplied excerpts.
Do not launch another full run on that evidence alone.

The mapper client now blocks for the **first task result**, verifies its NPZ
from the controller (shared storage check), and limits outstanding mapping
submissions to 16. It also consumes actual task results, surfacing cancellations,
timeouts and missing return values rather than relying only on a progress bar.
This addresses the submission burst; the scheduler log alone does not prove why
no checkpoints were written. Binning/splitting/reducing have not been changed.

Before another full import, submit a single compute job running:

```bash
uv run python -m scripts.jades.probe_hats_mapping \
  --scratch-dir "$JADES_SCRATCH" --probe-root "$JADES_RUN" --highest-order 12
```

Request 2 CPUs, 8G RAM and 15 minutes for this probe. It creates one local Dask
worker **in a separate process**, maps only one existing Parquet file, checks
package/mapper consistency and row count, then reads its checkpoint from the
client. No extraction/HATS outputs are changed; diagnostic files are kept in a
new `hats_mapping_probe_*` directory under `JADES_RUN`. A zero exit and `PASS`
establish that this one-file local-process path works, not that an external
multi-node cluster or the full import works. To test a live external cluster,
add `--scheduler-file "$JADES_DASK_FILE"` to this same probe before full import.

Recovery after run `7462528`: its controller reports 0/7600 map checkpoints,
but the available controller and worker excerpts contain no individual mapping
exception. The original storage/runtime cause is therefore not established.
The MMU importer now replaces row-count mapping tasks with a sparse counter:
it uses the installed reader and upstream coordinate-to-pixel iterator, writes
the same `indexes/counts/order` NPZ format atomically, and reads each checkpoint
back before returning success. Dask submits these file-producing tasks with
`pure=False` so a retry cannot reuse a cached successful result after a file
has disappeared. Other mapping modes and split/reduce/margin code are unchanged.
The sparse mapper avoids full-sky arrays per input file. Binning still uses
upstream dense arrays on the **client**, which needs more memory than the
previous 16 GiB request at high orders.

The measured preflight for this extraction is:
GOODS-S=303762, GOODS-N=180457, total=484219; maximum occupancy 1718 at order
12 and 485 at order 13. A recovery run can use **order 12 / threshold 2048**,
with **64G for the HATS client** and the existing eight 64G workers (56GiB
Dask limit). This reduces the dense sky footprint by four relative to order
13 without dropping or changing any selected source. Recheck with the cached
preflight using `--highest-order 12 --pixel-threshold 2048 --expected-rows
484219`. These changed parameters require a **new HATS output and staging
pair**: retain the old pair for diagnosis, reuse the existing extraction, and
do not bypass the old resume identity check. Always use a new scheduler JSON.

Local validation covers real HATS readers/checkpoints, a real Dask retry after
deleting a completed checkpoint, and an entire tiny collection import including
nested image values and margins. This is a tested mapping replacement, not a
claim that the unidentified Rusty failure has been reproduced or that all
cluster stages have been validated on the full dataset.

Start this stage **only after** every extraction task is complete, after
reviewing the image audit and confirming available Ceph space. HATS ingestion
can require substantial additional temporary space for partitioning and margin
creation. It is a separate Dask stage, not 32 independent HATS catalogs.

For the **first run**, choose a new output, staging, and scheduler filename,
and keep the HATS staging directory on shared Ceph so every worker can access
it. The ingest script now enables HATS checkpoints and one bounded retry of an
incomplete mapping stage. It also exposes the maximum HEALPix order and the
per-partition row threshold; defaults are order 13 / 1,024 rows. **Run the
preflight below** before accepting them. The upstream default of order 10 is
too coarse for dense JADES fields, while order 15 would require a dense
~96 GiB histogram for a single map, exceeding the old 12 GiB worker limit.
The importer checks that all 484,219 rows reach the main catalog.

If an older ingest failed **without** `--hats-resume`, its temporary checkpoints
were deleted: start with a fresh output and temp path. Never reuse an existing
partially written HATS output from an old non-resumable run.

Run the coordinate/partition audit first. It never modifies the extraction.
On the first run it reads only `field`, `object_id`, `ra`, `dec`, with bounded
parallel `ParquetFile` readers and `pre_buffer=False`, then saves a small atomic
coordinate cache beside the scratch directory. It avoids dataset-wide schema
discovery and background range coalescing across image-heavy row groups.
It reports progress every roughly 100 files, per-field checkpoint counts,
a selected-ID SHA256, maximum occupancy and theoretical *single-array* dense histogram
RAM for orders 10–13. The selected threshold must be at least as large as
the maximum occupancy at the selected finest order, and actual binning peak
RAM is higher than the histogram size alone:

```bash
uv run python -m scripts.jades.hats_healpix_preflight \
  --scratch-dir "$JADES_SCRATCH" \
  --raw-root "$JADES_RAW" --manifest "$JADES_MANIFEST" \
  --io-workers 8 \
  --highest-order 13 --pixel-threshold 1024 --expected-rows 484219
```

The default cache is `${JADES_SCRATCH}.preflight-coordinates.npz`. Subsequent
runs validate the complete file inventory, sizes/mtimes and completed extraction
checkpoints, then reuse it **without opening any Parquet files**. There are still
directory/stat calls on Ceph. Use `--coordinate-cache PATH.npz` to choose another
cache location, or `--refresh-cache` to force a fresh scalar scan. Cache validity
assumes ordinary immutable checkpoint files; after storage repair that preserves
both file size and mtime, explicitly refresh. A corrupt cache fails visibly;
refresh it to rebuild. No cutout images or extraction files are rewritten.

The two original FITS catalogs include sources rejected during cutout extraction.
Checkpoint counts/positions alone cannot identify those rejected sources. Thus
the first exact preflight still has to open the committed Parquet files; it must
not silently use all 485,510 parent sources or infer selection from a row count.
The 484,219 retained IDs and their original coordinates are preserved directly.
`--raw-root` additionally joins those field-qualified IDs to each FITS `SIZE`
table, verifies exact RA/Dec equality, and prints retained/excluded field counts.
Unknown/duplicate IDs, null/invalid coordinates, incomplete shard grids,
missing/extra Parquet files, and per-checkpoint row-count mismatches fail.
Both GOODS fields and every shard must be complete before this final preflight.

For Rusty, run within a compute allocation, for example
`srun --partition=genx --cpus-per-task=2 --mem=4G --time=01:00:00 --pty bash -l`,
then run the command above from the repository root. Eight I/O readers overlap
filesystem latency; they do not require eight CPUs. The first-run speed depends
on Ceph latency; no Rusty timing is claimed from local tests. To compare a cold
and warm run, prefix each with `time`; use `--refresh-cache` only for the cold run.

If the audit fails, **do not submit the HATS jobs**. Adjust the threshold or
order based on the printed occupancies and rerun the preflight. A higher order
increases RAM quickly (`~1.5 GiB` for order 12, `~6 GiB` for order 13, and
`~24 GiB` for order 14 *per dense histogram*, before intermediate arrays).
For order 13 we provision 64 GiB workers to leave room for the binning and
reduction stages; start with fewer workers if the scheduler has limited RAM.

```bash
export JADES_HATS_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export JADES_HATS_OUT="$JADES_RUN/hats_jades_jwst_${JADES_HATS_TAG}"
export JADES_HATS_TMP="$JADES_RUN/hats_tmp_jwst_${JADES_HATS_TAG}"
export JADES_DASK_FILE="$JADES_RUN/dask_jwst_${JADES_HATS_TAG}.json"
export JADES_DASK_WORKERS=8
export JADES_DASK_WORKER_MEMORY=56GiB
export JADES_HATS_MAX_ORDER=13
export JADES_HATS_PIXEL_THRESHOLD=1024
export JADES_HATS_MAPPING_RETRIES=1

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
  --mem=64G --time=24:00:00 \
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
find "$JADES_HATS_OUT" -maxdepth 4 -type f -name 'hats.properties'
uv run python - <<'PY'
import os
from pathlib import Path
import pyarrow.parquet as pq

main = Path(os.environ["JADES_HATS_OUT"]) / "jades_jwst" / "jades_jwst"
leaves = list((main / "dataset").rglob("Npix=*.parquet"))
rows = sum(pq.read_metadata(leaf).num_rows for leaf in leaves)
print(f"HATS main catalog: {len(leaves)} HEALPix partitions; {rows:,} rows")
assert rows == 484219, "HATS row count differs from extracted JADES objects"
assert (main / "hats.properties").exists(), "Missing main HATS metadata"
PY
```

The collection is named `jades_jwst`. The HATS inner catalog is typically
`$JADES_HATS_OUT/jades_jwst/jades_jwst` (with a margin catalog alongside it).
After verifying successful import, explicitly release the two long-running
Dask allocations with `scancel "$DASK_WORKER_JOB" "$DASK_SCHED_JOB"`.
The scheduler file may already disappear when Dask stops; always select a new
name on the next attempt.

**If HATS ingestion fails:** check `sacct` and the client log. A message like
`some map stages did not complete successfully` means that at least one of the
mapping histograms is missing; a 100% Dask progress bar only means that all
futures *finished*, not that all succeeded. The patched importer reports the
number and first 20 missing `map_N.npz` checkpoints and keeps the staging
directory on Ceph. A mapping retry can reuse already completed histogram files.

For a subsequent resume, keep **the same** `JADES_HATS_OUT` and
`JADES_HATS_TMP` (including the automatically written
`hats_import_jades_jwst_resume/_mmu_resume_identity.json`), but select a
**new** `JADES_DASK_FILE` and start a new scheduler, worker array, and HATS
client with the commands above. The resumed run checks input paths, catalog
name, output path and import parameters against its stored fingerprint before
touching any partial output. Do not edit or delete the checkpoint directory.
Never have two HATS client jobs writing to the same output/staging pair.

To watch mapping checkpoints while the map stage runs:

```bash
find "$JADES_HATS_TMP/hats_import_jades_jwst_resume" -name 'map_*.npz' | wc -l
tail -n 40 "$JADES_RUN/logs/hats_${HATS_JOB}.out"
```

Mapping histograms can disappear after a *successful* mapping stage, when the
pipeline advances to binning. Do not interpret a later zero count as failure.

**Important constraints:** Slurm partitions/accounts/wall-time limits are
site-specific; adjust these resource requests if Rusty rejects them. Scale
the number of *concurrent* workers gradually because Ceph I/O can saturate.
An interrupted extraction is resumable at committed Parquet checkpoints;
with the updated import script, HATS is also resumable when its source inputs,
configuration and persistent temporary directory are unchanged.
