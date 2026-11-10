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

## Scientific sources

- [Official release and attribution](https://jades-survey.github.io/scientists/data.html)
- [DR5 NIRCam mosaics, Johnson et al. 2026](https://arxiv.org/abs/2601.15954)
- [DR5 photometric catalogue, Robertson et al. 2026](https://arxiv.org/abs/2601.15956)
- [Temporary UCSC archive](https://slate.ucsc.edu/~brant/jades-dr5/)

This is a **manifest-pinned** downloader. It intentionally does not scrape an
unversioned directory and guess which of the many available coadds, epochs,
PSF-homogenized mosaics, or subregions to use. The first manifest must be
populated after inspecting the real DR5 FITS headers. Subsequent runs are
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

Different files can be listed for non-overlapping footprints of the same
field/band. The builder currently rejects overlapping coverage implicitly by
requiring a single selected image per object/band; a manifest containing
overlapping products should be deduplicated during manifest review.

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

Run on the Flatiron/JADES-enabled processing environment with MMU v2
dependencies installed. The first command downloads the pinned inputs;
partial downloads resume if the HTTP server supports range requests.

```bash
python -m scripts.jades.download_data \
  --manifest /path/to/jades_dr5_manifest.json \
  --raw-root /mnt/ceph/users/polymathic/external_data/astro/JADES_DR5

python -m scripts.jades.audit_catalog \
  /path/to/GOODS-S.fits /path/to/GOODS-N.fits \
  --bands F090W F115W F150W F200W F277W F356W F410M F444W \
  --mag-band F277W --mag-cuts 26 27 28 29 30 \
  --output /path/to/jades_snr_audit.json

python -m scripts.jades.build_parent_sample_hats \
  --instrument jwst \
  --manifest /path/to/jades_dr5_manifest.json \
  --raw-root /mnt/ceph/users/polymathic/external_data/astro/JADES_DR5 \
  --scratch-dir /mnt/ceph/users/polymathic/jades_jwst_hats_scratch \
  --download --max-objects 1000

python -m scripts.jades.build_parent_sample_hats \
  --instrument hst \
  --manifest /path/to/jades_dr5_manifest.json \
  --raw-root /mnt/ceph/users/polymathic/external_data/astro/JADES_DR5 \
  --scratch-dir /mnt/ceph/users/polymathic/jades_hst_hats_scratch \
  --max-objects 1000
```

Omit `--max-objects` for the full build. **Use a separate scratch directory
for each sample/configuration**: completion markers are used for restart.
Do not run a smoke-test and full production in the same scratch directory.

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

No full DR5 end-to-end run has been claimed: source download availability and
instrument-specific FITS headers must first be verified on real files.
