# MMU v2 HATS — Port Status

Live tracker for the raw → HATS port of v1 MMU. Every non-skipped dataset in `mmu/hats_configs.py` gets a `scripts/<dataset>/build_parent_sample_hats.py` that matches v1's HuggingFace `Features(...)` schema 1:1, with HATS output instead of HDF5.

**Test slice:** COSMOS field — RA=150°, Dec=+2°, radius=0.5° (1° cone). Every build script accepts `--ra-center/--dec-center/--radius`. The `cosmos` Snakemake profile sets these automatically.

**Schema rule:** Struct-of-parallel-lists with plain pyarrow types. NO HF `datasets` extension types inside `list<>` or `struct<>` (see `project_image_storage` memory — nested_pandas crashes on them in hats-import's finishing stage).

**No normalization:** Every catalog keeps its survey-native column names, units, and values untouched. No canonical aliases, no unit conversions, no cross-survey unification. Users who want a unified view build it on top.

**Scatter/gather sharding:** Large datasets are built via scatter-gather — N independent sbatch jobs each process a stride slice of the inputs and write per-unit parquet shards to a shared ceph scratch dir, then one gather job runs `write_hats_from_parquet_dir`. Controlled by `SHARDED_DATASETS` in the Snakefile. All dependency tracking goes through Snakemake's DAG, no bespoke shell launchers. Scatter jobs run on `preempt` (bursts into idle rome nodes cluster-wide); gather jobs run on `ccm` (guaranteed, non-preemptible so a half-written HATS catalog is impossible).

**Legend:** `[x]` done · `[ ]` pending · `[~]` in progress · `[skip]` out of scope

---

## Production builds landed so far

| Dataset | Rows | Size | Sky frac | Notes |
|---|---:|---:|---:|---|
| sdss | 4,254,830 | 193 GB | 0.35 | DR17 spectra, full run |
| gaia (XP subset) | 219,197,643 | 293 GB | 1.00 | GaiaSource × XpContinuousMeanSpectrum join. See TODO below — user wants `gaia` renamed to `gaia_xp` and a NEW `gaia` containing all ~1.8B DR3 sources. |
| galex | 82,992,062 | 18 GB | 0.82 | GUVCat AIS |
| tess | 159,994 | 17 GB | 0.06 | SPOC FFI lightcurves |
| sages | 29,332,961 | 1.5 GB | 0.26 | DR1 u/v photometry |

**Total landed so far: 336 M rows, 523 GB.**

---

## In flight right now

- [~] **legacysurvey** — 128-shard preempt scatter running. Previous run had a ChunkedArray bug that silently dropped 135 large sweeps; fix landed in commit `a81eae9`, skip-if-exists optimization in `a880b99` lets the rerun reuse the 211 good parquets already on disk.
- [~] **desi** — scatter complete (16/16 markers exist), gather rerun pending with `ingest_workers=8` after the first attempt crashed in the hats-import splitting stage with too-many-dask-workers OOM.
- [~] **manga** — scatter complete (8/8 markers), gather rerun pending with `ingest_workers=8` (same reason as desi; first attempt crawled to 2% in 1h43m with constant dask worker restarts).
- [~] **foundation / snls / ps1_sne_ia / des_y3_sne_ia / swift_sne_ia** — all 5 SN-Ia scripts queued in the current snakemake master, tiny datasets (100–400 SNe each), expected ~1 min wall each.

## Ported + tests passing (baseline)

- [x] **sdss** — spectra, scatter/gather sharded (20 shards in speedster run)
- [x] **desi** — spectra with fork-COW catalog sharing in scatter workers
- [x] **ssl_legacysurvey** — image, struct-of-parallel-lists reference impl
- [x] **tess** — timeseries, batched streaming + Pool(96)
- [x] **allwise** — tabular, healpix-k5 prefilter + Pool(96)
- [x] **twomass** — tabular, per-shard streaming Pool
- [x] **galex** — tabular, per-FITS-shard streaming
- [x] **sages** — tabular, single FITS
- [x] **gaia** — BUNDLED (XP + photometry + 20-field astrometry + RV + gspphot), scatter/gather sharded
- [x] **legacysurvey** — BUNDLED (images + RGB + masks + nearby catalog), scatter/gather sharded
- [x] **manga** — BUNDLED IFU cubes + griz images + DAP maps, scatter/gather sharded
- [x] **foundation / snls / ps1_sne_ia / des_y3_sne_ia / swift_sne_ia** — SNANA ASCII lightcurves via shared `mmu.sn_ia_snana` helper

**Total: 16 datasets with working scripts + tests green.**

## Phase 2 — Remaining datasets to port

### Spectra (reuse sdss/desi pattern)
- [ ] **vipers** — VIMOS Public Extragalactic Redshift Survey
- [ ] **galah** — GALactic Archaeology with HERMES
- [ ] **apogee** — Near-IR stellar spectra with pseudo_continuum
- [ ] **chandra** — X-ray spectra (ene_low/high/center bins)

### Images (reuse ssl_legacysurvey pattern)
- [ ] **jwst** — BUNDLED: NIRCam multi-band + metadata
- [ ] **btsbot** — ZTF BTS image triplets
- [ ] **gz10** — Galaxy Zoo 10 single RGB + classification label

### Tabular
- [ ] **desi_provabgs** — Stellar parameters + MCMC posteriors (100×13 samples per object)

## Follow-up work

- [ ] **gaia restructure**: split the current 220M XP-joined `gaia` catalog into
  - `gaia` — ALL ~1.8B Gaia DR3 sources from `GaiaSource_*.hdf5`, no XP filter. New build script.
  - `gaia_xp` — the current 220M XP-joined catalog, renamed from `gaia`.
  Means moving the existing catalog and adding a second entry in `mmu/hats_configs.py`. Queued for after the current production run settles.
- [ ] **ssl_legacysurvey full-sky production**: ~42 TB expected output, crosses the Flatiron ceph "let us know if you'll generate >10 TB" courtesy threshold. Need to email scicomp before launching.

## Skipped (out of scope)

- [skip] **plasticc** — raw dir empty on cluster
- [skip] **kepler** — raw FITS not mirrored; only v1-processed HDF5 exists
- [skip] **hsc** — source catalog not on cluster mirror
- [skip] **cfa**, **csp**, **lamost**, **yse** — in v1 but deliberately excluded

## Infrastructure shipped

- `mmu/cone.py` — shared cone-cut helper (haversine + bbox prefilter)
- `mmu/safety.py` — output-path guardrails (allowlist + `external_data/` denylist) with 30 tests
- `mmu/hats_import.py` — `write_hats` and `write_hats_from_parquet_dir` helpers
  - Production defaults: `debug=False, n_workers=32` (previously `debug=True, 1` was silently bottlenecking ingest to single-core)
  - `default_scratch_dir()` for per-catalog ceph scratch paths
- `mmu/sn_ia_snana.py` — shared build helper for the 5 SNANA ASCII SN-Ia datasets
- `mmu/hats_configs.py` — registry with `DatasetConfig` + raw paths
- `Snakefile`
  - Per-dataset `build_<name>` rules, each declaring its build script as an `input:` (auto-rerun on edit)
  - `SHARDED_DATASETS` config + generic `build_sharded_shard` wildcard rule + loop-generated gather rules
  - `SHARDED_SLURM_PARTITION` for scatter (preempt), gather always runs on `SLURM_PARTITION` (ccm, guaranteed)
  - `validate_hats_root` top-level safety check
  - `qos="preempt"` resource for preempt scatter, `--slurm-requeue` auto-requeue on preemption
- `snakemake_config.yaml` — `cluster`, `cosmos`, `test` profiles

## SLURM launchers

- `scripts/slurm/run_cosmos_slurm.sh` — COSMOS 1° validation via Snakemake/slurm executor
- `scripts/slurm/run_production_slurm.sh` — all-sky production via Snakemake/slurm executor

No more bespoke `/tmp/*.sh` launchers — everything flows through Snakemake now, including scatter/gather sharded builds.

---

## Totals

| Category | Count |
|---|---|
| Ported + tests green | 16 |
| Landed in production | 5 (sdss, gaia[-xp], galex, tess, sages) |
| In flight right now | 8 (legacysurvey, desi, manga, foundation, snls, ps1_sne_ia, des_y3_sne_ia, swift_sne_ia) |
| Phase 2 remaining to port | 8 (vipers, galah, apogee, chandra, jwst, btsbot, gz10, desi_provabgs) |
| Follow-up | 2 (gaia restructure, ssl_legacysurvey full-sky) |
| Skipped | 7 (plasticc, kepler, hsc, cfa, csp, lamost, yse) |

## Verification criteria (per dataset)

1. `uv run pytest tests/test_<dataset>_build.py -v` → all pass
2. `uv run pytest tests/` → full suite green
3. Cluster COSMOS slice or full production build produces a valid HATS catalog with `hats.properties`
4. Read-back parquet matches v1 HF schema field-for-field
5. `lsdb.read_hats(<output>).compute().head()` → returns rows without exceptions
6. Checkbox ticked here
