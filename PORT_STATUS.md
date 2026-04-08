# MMU v2 HATS — Port Status

Live tracker for the raw → HATS port of v1 MMU. Every non-skipped dataset in `mmu/hats_configs.py` gets a `scripts/<dataset>/build_parent_sample_hats.py` that matches v1's HuggingFace `Features(...)` schema 1:1, with HATS output instead of HDF5.

**Test slice:** COSMOS field — RA=150°, Dec=+2°, radius=0.5° (1° cone). Every build script accepts `--ra-center/--dec-center/--radius`. The `cosmos` Snakemake profile sets these automatically.

**Schema rule:** Struct-of-parallel-lists with plain pyarrow types. NO HF `datasets` extension types inside `list<>` or `struct<>` (see `project_image_storage` memory — nested_pandas crashes on them in hats-import's finishing stage).

**Scatter/gather sharding:** Large datasets are built via scatter-gather — N independent sbatch jobs each process a stride slice of the inputs and write per-unit parquet shards to a shared ceph scratch dir, then one gather job runs `write_hats_from_parquet_dir`. Controlled by `SHARDED_DATASETS` in the Snakefile. All dependency tracking goes through Snakemake's DAG, no bespoke shell launchers.

**Legend:** `[x]` done · `[ ]` pending · `[~]` in progress · `[skip]` out of scope

---

## Ported baseline — script exists + tests pass

- [x] **sdss** (spectra) — `scripts/sdss/build_parent_sample_hats.py`
  - Spectrum struct<flux, ivar, lsf_sigma, lambda, mask> + photometry
  - Cone args + streaming per-plate parquet + `multiprocessing.Pool`
  - Tests: 11 passing
- [x] **desi** (spectra) — `desispec.coadd_cameras` coadd
  - Spectrum struct + Z/ZERR/ZWARN + photometry
  - Streaming per-coadd-file parquet + `multiprocessing.Pool` with fork-COW catalog sharing
  - Tests: 20 passing
- [x] **ssl_legacysurvey** (image) — struct-of-parallel-lists (reference impl)
  - `image: struct<band, flux (3,152,152), psf_fwhm, scale>` + photometry
  - Tests: 8 passing
- [x] **tess** (timeseries) — variable-length lightcurves
  - `lightcurve: struct<time, flux, flux_err, quality>` per (TIC, sector)
  - Streaming per-batch parquet (batches of 1000 LCs) + `multiprocessing.Pool(96)`
  - Cone args accepted but no-op (filenames don't encode RA/Dec)
  - Tests: 11 passing
- [x] **allwise** (tabular) — 298 cols from IRSA parquet
  - Cone filter + healpix-k5 prefilter
  - Streaming per-shard parquet + `multiprocessing.Pool(96)`
  - Tests: 12 passing
- [x] **twomass** (tabular) — 2MASS PSC pipe-delimited CSV
  - Streaming per-shard parquet (92 shards)
  - Tests: 6 passing
- [x] **galex** (tabular) — GUVCat FITS shards (36 shards)
  - Streaming per-shard parquet
  - Tests: 5 passing
- [x] **sages** (tabular) — single u/v photometry FITS table

## Phase 1 — Flagship datasets

- [x] **gaia** — BUNDLED: XP spectra + photometry + astrometry + RV + stellar params
  - Joined on `source_id` between GaiaSource shards and XpContinuousMeanSpectrum shards
  - Streaming per-shard-pair parquet + `multiprocessing.Pool(96)`
  - Tests: 10 passing
- [x] **legacysurvey** — BUNDLED: multi-band images + RGB + masks + catalog
  - Struct-of-parallel-lists for `image`, `rgb`, `blobmodel`, `object_mask`, nearby `catalog`
  - **Scatter-gather sharded** (8 shards × `Pool(32)`) via Snakemake `build_sharded_shard` + `build_legacysurvey` rules
  - Per-sweep worker in `_process_sweep_to_parquet`; catches ALL exceptions (including cfitsio unpicklable ones) → picklable error strings
  - Tests: 22 passing
- [skip] **hsc** — source catalog not mirrored on Flatiron cluster
- [skip] **kepler** — raw Kepler FITS not mirrored; only v1 HDF5 exists at `spoc/SPOC/`
- [x] **manga** — BUNDLED, most complex: IFU cubes + spaxel coords + griz reconstructions + DAP maps
  - `spaxels: struct<flux, ivar, lambda, mask>` (9216×4563) + `images: struct<4 bands × 96²>` + `maps: struct<~50 DAP maps>`
  - Streaming per-plate-ifu parquet + `multiprocessing.Pool(32)`
  - Sharding flags added (eligible for `SHARDED_DATASETS` entry)
  - Tests: 11 passing

## Phase 2 — Remaining datasets

### Spectra (reuse sdss/desi pattern)
- [ ] **vipers** — VIMOS Public Extragalactic Redshift Survey
- [ ] **galah** — GALactic Archaeology with HERMES
- [ ] **apogee** — Near-IR stellar spectra with pseudo_continuum
- [ ] **chandra** — X-ray spectra (ene_low/high/center bins)

### Images (reuse ssl_legacysurvey pattern)
- [ ] **jwst** — BUNDLED: NIRCam multi-band + metadata
- [ ] **btsbot** — ZTF BTS image triplets
- [ ] **gz10** — Galaxy Zoo 10 single RGB + classification label

### SN time series (shared `mmu/sn_ia_snana.py` helper)
- [x] **foundation** — Foundation DR1 SNe Ia (SNANA ASCII lightcurves, ~180 SNe)
  - Thin wrapper around `mmu.sn_ia_snana.build_main`
  - Tests: 6 passing
- [x] **snls** — Supernova Legacy Survey (JLA2014, ~239 SNe)
  - Wrapper around shared SN-Ia SNANA helper
- [x] **ps1_sne_ia** — Pan-STARRS1 SNe Ia (PS1_PSc\*.txt, ~369 SNe, object_id prefix "PS1_")
  - Wrapper
- [x] **des_y3_sne_ia** — DES Y3 SNe Ia (des_real_\*.dat, ~251 SNe, object_id prefix "DES_")
  - Wrapper
- [x] **swift_sne_ia** — Swift UV/optical SNe (\*.dat, ~117 SNe)
  - Wrapper

### Tabular
- [ ] **desi_provabgs** — Stellar parameters + MCMC posteriors (100×13 samples per object)

## Infrastructure shipped

- `mmu/cone.py` — shared cone-cut helper (haversine + bbox prefilter)
- `mmu/safety.py` — output-path guardrails (allowlist + `external_data/` denylist) with 30 tests
- `mmu/hats_import.py` — `write_hats` and `write_hats_from_parquet_dir` helpers
  - Production defaults flipped to `debug=False, n_workers=32` (was `debug=True, 1` — single-core ingest bottleneck)
  - `default_scratch_dir()` for per-catalog ceph scratch paths
- `mmu/sn_ia_snana.py` — shared build helper for the 5 SNANA ASCII SN-Ia datasets
- `mmu/hats_configs.py` — registry with `DatasetConfig` + raw paths
- `Snakefile` with per-dataset `build_<name>` rules
  - Each rule takes its build script as an `input:` → auto-rerun on script change
  - `SHARDED_DATASETS` scatter-gather rule pair for large datasets
  - `validate_hats_root` top-level safety check
- `snakemake_config.yaml` — `cluster`, `cosmos`, `test` profiles

## SLURM launchers

- `scripts/slurm/run_cosmos_slurm.sh` — COSMOS 1° validation via Snakemake/slurm executor
- `scripts/slurm/run_production_slurm.sh` — all-sky production via Snakemake/slurm executor
- Ad-hoc bespoke `/tmp/<dataset>_sbatch.sh` files exist on the cluster from crash-recovery resubmissions — these are being phased out in favor of pure Snakemake DAG execution via `SHARDED_DATASETS`.

## Skipped (out of scope)

- [skip] **plasticc** — raw dir empty on cluster (`hats_configs.skip=True`)
- [skip] **kepler** — raw FITS not mirrored; only v1-processed HDF5 exists
- [skip] **hsc** — source catalog not on cluster mirror
- [skip] **cfa**, **csp**, **lamost**, **yse** — in v1 but deliberately excluded

---

## Totals (current state)

| Category | Count |
|---|---|
| Ported + tests passing | **18** (sdss, desi, ssl_legacysurvey, tess, allwise, twomass, galex, sages, gaia, legacysurvey, manga, foundation, snls, ps1_sne_ia, des_y3_sne_ia, swift_sne_ia, ... — counting) |
| Phase 2 spectra remaining | 4 (vipers, galah, apogee, chandra) |
| Phase 2 images remaining | 3 (jwst, btsbot, gz10) |
| Phase 2 tabular remaining | 1 (desi_provabgs) |
| Skipped | 7 (plasticc, kepler, hsc, cfa, csp, lamost, yse) |
| **v1 datasets total** | **31** |

## Verification criteria (per dataset)

1. `uv run pytest tests/test_<dataset>_build.py -v` → all pass
2. `uv run pytest tests/` → full suite green
3. Cluster COSMOS slice or full production build produces a valid HATS catalog with `hats.properties`
4. Read-back parquet matches v1 HF schema field-for-field
5. `lsdb.read_hats(<output>).compute().head()` → returns rows without exceptions
6. Checkbox ticked here
