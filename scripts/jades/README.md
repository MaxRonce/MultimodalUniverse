# JADES DR5 → MMU v2 HATS

This builder targets **JADES DR5 photometric catalogues and imaging mosaics** in
GOODS-N and GOODS-S. It produces two separate MMU datasets:

- `jades_jwst`: 18 JWST/NIRCam + 8 JWST/MIRI filters (as available).
- `jades_hst`: 5 HST/ACS + 4 HST/WFC3 filters (as available).

They have **independent magnitude-selection and image-size settings**, output directories,
and future Hugging Face publishing targets. The current scripts build HATS,
but **do not publish to Hugging Face automatically**.

It uses the MMU v2 `image` struct and the shared
`mmu.hats_import.write_hats_from_parquet_dir` utility. It does **not** modify
the legacy JWST/DJA builder. These datasets may overlap on the sky; do not
assume that their object samples are independent.

**Current Rusty runbook:** [RUSTY_RUNBOOK.md](RUSTY_RUNBOOK.md). The runnable
JWST path uses the 34 NIRCam v5.0 + 18 MIRI v1.0 images downloaded for this
project, along with the two GOODS-S/GOODS-N photometric catalogs. The HST
builder remains separate and requires HST images not yet obtained here.

## Scientific sources

- [Official release and attribution](https://jades-survey.github.io/scientists/data.html)
- [DR5 NIRCam mosaics, Johnson et al. 2026](https://arxiv.org/abs/2601.15954)
- [DR5 photometric catalogue, Robertson et al. 2026](https://arxiv.org/abs/2601.15956)
- [Temporary UCSC archive](https://slate.ucsc.edu/~brant/jades-dr5/)

This is a **manifest-pinned** downloader. It intentionally does not scrape an
unversioned directory and guess which of the many available coadds, epochs,
PSF-homogenized mosaics, or subregions to use. `make_manifest.py` validates
local FITS headers and pins the precise downloaded NIRCam/MIRI images.
Subsequent runs are
non-interactive, resumable, and reproducible. Prefer permanent MAST URLs as
soon as their DR5 products become available.

## Manifest schema

```json
{
  "release": "JADES_DR5",
  "files": [
    {"path": "catalogs/GOODS-S.fits", "url": "https://EXACT-ARCHIVE-URL", "sha256": "OPTIONAL-SHA256"},
    {"path": "mosaics/GOODS-S-F277W.fits", "url": "https://EXACT-ARCHIVE-URL"}
  ],
  "catalogs": [{"field": "GOODS-S", "path": "catalogs/GOODS-S.fits"}],
  "mosaics": [{
    "field": "GOODS-S", "band": "F277W", "path": "mosaics/GOODS-S-F277W.fits",
    "sci_hdu": "SCI", "uncertainty_hdu": "ERR", "uncertainty_kind": "sigma",
    "unit": "MJy/sr"
  }]
}
```

**Do not use the example URLs or HDU names without checking the release.**
Every `files[].path` must correspond to a catalogue or mosaic `path` entry.
`sci_hdu` and `uncertainty_hdu` may also be integer FITS extension indices.
The uncertainty kind is mandatory: `sigma` (ERR), `variance`, or `ivar`
(genuine inverse variance). `WHT` does **not** automatically imply inverse
variance: inspect the specific product's definition. If SCI and uncertainty
are stored in different FITS files, they must first be packaged or the builder
extended; this is deliberately not guessed.

Different files can be listed for overlapping footprints of the same
field/band. The builder selects exactly one per source/band, prioritizing
valid pixel count and then local inverse variance, and records the original
image name in the `image_sources` list. MIRI photometry comes from the
catalog's `MIRI` HDU, not its `CIRC` HDU.

For the downloaded JADES DR5 mosaics, create the manifest with:

```bash
uv run python -m scripts.jades.make_manifest \
  --raw-root /mnt/ceph/users/polymathic/external_data/Astro/JADES_DR5 \
  --output /mnt/ceph/users/polymathic/mronceray/jades_dr5_runs/jades_dr5_manifest.json
```

This checks the 52 mosaics and two catalogs, with `SCI`, `ERR`, WCS, dimensions
and MJy/sr pixel area. The MIRI images are the [JADES DR5 coordinated parallels
v1.0](https://github.com/staceyalberts/JWST-JADES-MIRI-PARALLELS/releases/tag/v1.0)
covering F770W, F1280W and F1500W. `ERR` is a per-pixel resampled
uncertainty, not a fully correlated-noise-aware aperture uncertainty.

## Commands

### Download all NIRCam DR5 mosaics with live progress

The standalone downloader handles the 34 global NIRCam GOODS-S/GOODS-N
mosaics with **four simultaneous transfers** by default. Every 10 seconds it
prints total bytes, global percentage, aggregate transfer rate, estimated time
remaining, and progress of each active band. Each transfer has its own log.
Use `--jobs 8` for eight concurrent transfers. Each connection is capped at
20 MiB/s, keeping the aggregate theoretical ceiling below 200 MB/s.
It resumes existing `.fits` files from previous `wget -c` runs, compares each
final size to HTTP `Content-Length`, and reports success only after all 34
are complete. It does **not** fetch MIRI or HST products.

```bash
bash scripts/jades/download_nircam_mosaics.sh --jobs 4
# To override locations:
# RAW=/path/to/JADES_DR5 LOG=/path/to/logs bash scripts/jades/download_nircam_mosaics.sh
```

Run it in `tmux` or another persistent session. The current HTTP mirror is
not authenticated; verify provider-supplied checksums when available. A size
match alone is not a cryptographic integrity check.

### Manifest-pinned downloader

See the [exact Rusty commands](RUSTY_RUNBOOK.md) for manifest generation,
NIRCam- and MIRI-focused smoke tests, validation/preview PNGs, HATS pilot,
catalog SNR/magnitude audits, production scaling and crash recovery.
`download_data.py` remains available to fetch pinned inputs with `--download`
when raw files are missing, but the current workflow starts from locally
downloaded and inspected data. Do not try to build `jades_hst` using the JWST
manifest alone: it contains no HST mosaics.

Omit `--max-objects` for the full build. **Use a separate scratch directory
for each sample/configuration**: restartable per-shard Parquet progress files
and completed-field markers validate the configuration and skip already
finished work. Do not run a smoke-test and full production in the same
scratch directory. For current memory-bounded defaults, `--chunk-size 4`
serializes at most four records, `--rows-per-file 64` sets checkpoint shard
size, `--hats-batch-size 4` bounds HATS import reads, and `--skip-hats` allows
Parquet-only QC before the costly HATS stage. `--sample-mode miri` targets
catalog sources with MIRI exposure for a coverage smoke test.

Each build also accepts its own `--image-size` (native pixels). Choose this
after comparing angular footprint and pixel scale across JWST and HST; keeping
the same pixel count does *not* guarantee the same angular field of view.

Once the scientific completeness analysis has selected independent magnitude
limits, add `--mag-band F277W --mag-max <JWST_LIMIT>` for JWST and, for
example, `--mag-band F160W --mag-max <HST_LIMIT>` for HST. This computes
AB apparent magnitudes from **positive KRON fluxes** (nJy) using
`m_AB = 31.4 - 2.5 log10(flux_nJy)`. The reference bands and magnitude limits
are **not prescribed by JADES** and must be justified independently for both
instruments. A reference band absent from a source means it fails that cut.

## Selection and schema

- No global SNR cut. The JADES source detection algorithm already used
  multiband SNR images; catalog-level thresholds `2, 3, 5, 10` are reported
  separately by the audit, using `CIRC3 / CIRC3_e` (r=0.25 arcsec) and 1/2/4
  detected bands. The paper's segmentation thresholds must **not** be
  interpreted as object-level photometric cuts.
- Two independent optional apparent-magnitude cuts are supported, using the
  relevant reference-band `KRON` flux. They are intentionally unset until
  catalogue counts and physical completeness are assessed.
- Include all available bands (up to 35 in DR5) with **real pixel-level
  inverse variance**. For a given source, absent bands are omitted from the
  variable-length `image.band` list. A source without any valid band is skipped.
- Images are stored as `image.{band, flux, ivar, mask, scale}`. `flux` is
  nJy/pixel, `ivar` is nJy^-2/pixel² (inverse variance of pixel flux),
  `mask=True` means a finite science pixel and strictly positive real weight.
  Invalid pixels get `flux=0`, `ivar=0`, `mask=False`.
- The matching `photometry` struct stores the catalogue CIRC3 flux and
  **correlated-noise-aware** `CIRC3_e` error in nJy. These are *aperture*
  uncertainties, not a sum of independent pixel inverse variances.
- Object IDs include the GOODS field to prevent collisions. Source positions
  come from the DR5 `SIZE` HDU; shape features `A` and `B` are retained.
- Each science band carries its own WCS/scale. Cutouts are 160×160 **native
  pixels**, so different instruments can cover different angular extents.
  Coordinate transformations are vectorized per mosaic before extraction.
- Original DR5 `PHOTOZ` fields `z_spec`, `z_peak`, `z_ml`, `l68`, `u68` are
  preserved as a separate `redshift` struct, with invalid negatives as NaN.
- NIRCam/HST aperture photometry comes from `CIRC`, but MIRI from `MIRI`.
  Photometry lists align with the selected image bands; chosen source FITS
  paths align through the `image_sources` list.
- Only `nJy`, `Jy` and `MJy/sr` science units are supported. `MJy/sr` must have
  the pixel solid angle `PIXAR_SR`; otherwise the build fails rather than
  assuming a nominal pixel scale.

## Required validation before production

1. Verify the manifest against the actual FITS archive and pin checksums.
2. Verify **every** selected band's `SCI` and uncertainty HDU semantics,
   shape, BUNIT and pixel area, especially HST vs NIRCam vs MIRI.
3. Compare cutout aperture sums with the corresponding JADES catalogue values
   after accounting for the exact aperture and PSF corrections.
4. Report object counts, band availability, invalid pixel fractions, SNR
   histograms, duplicate IDs and all missing/degraded subregions.
5. Evaluate PSF representation and native-vs-common-PSF products before
   training; this builder does not yet add spatially varying PSF images.

No full DR5 end-to-end HATS run has been claimed. Header structure has been
inspected on Rusty, but scientific cross-validation and production-scale
performance/disk capacity still need to be measured. Full 160×160 multi-band
cutouts can require **terabytes** of scratch and output, unlike tiny row-only
astronomical catalogs.
