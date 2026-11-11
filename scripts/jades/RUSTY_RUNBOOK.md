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
```

Run long builds in `tmux new -s jades-build`. Reattach later with
`tmux attach -t jades-build`; detach with `Ctrl+B`, then `D`.

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

## 3. Run a tiny Parquet-only smoke test

The limit is **per GOODS field** (100 for GOODS-S and 100 for GOODS-N).
Do not reuse this scratch directory for a larger/different selection.

```bash
uv run python -m scripts.jades.build_parent_sample_hats \
  --instrument jwst \
  --manifest "$JADES_RUN/jades_dr5_manifest.json" \
  --raw-root "$JADES_RAW" \
  --scratch-dir "$JADES_RUN/scratch_smoke100" \
  --image-size 160 --max-objects 100 \
  --chunk-size 4 --rows-per-file 32 --skip-hats
```

To check **MIRI coverage specifically**, use a **different scratch**:

```bash
uv run python -m scripts.jades.build_parent_sample_hats \
  --instrument jwst \
  --manifest "$JADES_RUN/jades_dr5_manifest.json" \
  --raw-root "$JADES_RAW" \
  --scratch-dir "$JADES_RUN/scratch_miri100" \
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
  --scratch-dir "$JADES_RUN/scratch_smoke100" \
  --output "$JADES_RUN/smoke100_audit.json" \
  --max-samples 24 \
  --preview-dir "$JADES_RUN/smoke100_previews"
```

For the MIRI-focused sample:

```bash
uv run --extra viz python -m scripts.jades.inspect_cutouts \
  --scratch-dir "$JADES_RUN/scratch_miri100" \
  --output "$JADES_RUN/miri100_audit.json" \
  --max-samples 24 \
  --require-any-band F770W F1280W F1500W \
  --preview-dir "$JADES_RUN/miri100_previews"
```

The audit checks 2-D shapes, non-finite pixels, positive IVAR only where the
mask is true, unique band names, and photometry/provenance alignment. It also
reports actual row counts and per-band valid-pixel fractions in the sampled
objects. Preview RGB is display-scaled and not a scientific flux measurement.

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
  --scratch-dir "$JADES_RUN/scratch_smoke100" \
  --output-root "$JADES_RUN/hats_smoke100" \
  --tmp-dir "$JADES_RUN/hats_tmp" \
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
du -sh "$JADES_RUN/scratch_smoke100" "$JADES_RUN/hats_smoke100"
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
  --scratch-dir "$JADES_RUN/scratch_pilot1000" \
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
is an error by design. Do not manually edit `.progress.json` or `.done` files.

### What is intentionally not validated by these scripts

- The true end-to-end photometric calibration/PSF model for every filter;
- The science-level treatment of *correlated* noise in mosaic pixels;
- Full-depth completeness, morphology or photo-z selection;
- Full build runtime and final dataset size on Rusty (must be measured);
- HST mosaic production (a separate `jades_hst` workflow is pending).

For MIRI, `MIRI` catalog `FxxxW_CIRC3` and `FxxxW_CIRC3_e` are used instead
of the NIRCam/HST `CIRC` HDU; `PHOTOZ` provides `z_spec`, `z_peak`, `z_ml`,
`l68`, `u68`. The selected input mosaic for each band is preserved in the
top-level `image_sources` list, aligned to `image.band`.
