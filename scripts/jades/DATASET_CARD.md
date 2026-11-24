---
pretty_name: JADES DR5 JWST — MMU HATS
language:
  - en
tags:
  - astronomy
  - astrophysics
  - jwst
  - jades
  - multimodal
  - hats
  - aion-2
size_categories:
  - 100K<n<1M
---

# JADES DR5 JWST — Multimodal Universe

**484,219 selected sources** in GOODS-S and GOODS-N, with JWST image cutouts,
inverse variance, validity masks, photometry and available redshift information.
This is a derived **HATS collection**, intended for AION-2 training research.
Model training and end-to-end AION-2 loader validation are not established by this release.

**Release status:** private preparation; public redistribution terms for the exact
DR5 inputs still require confirmation. No dataset licence is granted by this card.
See “Licensing and attribution” below before making the repository public.

## Contents and population

| Property | Value |
| --- | --- |
| Main catalogue | `jades_jwst/` |
| Unique main sources | 484,219 |
| GOODS-S / GOODS-N | 303,762 / 180,457 |
| Main HATS partitions | 527 |
| Image size | 160 × 160 native pixels per available channel |
| Observations | JWST NIRCam and downloaded MIRI mosaics |
| Format | HATS spatial Parquet catalogue plus associated margin |
| Explicit train / validation / test splits | None |

The root `collection.properties` references the main catalogue and its margin.
Each catalogue has its own `hats.properties`. Preserve the entire directory tree.
The margin contains replicated neighbouring sources for spatial queries;
**do not count it as additional independent training data**.
`release/collection_files.json` records the actual files and their sizes;
`release/provenance.json` records validation and the serialized collection size.

## Inputs and selection

The upstream DR5 parent photometric catalogues contain 485,510 rows. This export
retains 484,219 sources: 604 GOODS-S and 687 GOODS-N rows are excluded by the
extraction validity conditions. It applies no additional magnitude, S/N,
redshift, star/galaxy or object-quality cut for this full JWST run.

Coordinates must be finite. Where a band's exposure column is present, its
exposure must be finite and positive. A band requires a finite science value
and positive inverse variance at the cutout centre. A source requires at least
one valid band. Available bands vary by source. For overlapping mosaics of the
same band, the extraction chooses by valid-pixel count, then median inverse
variance. Native pixel scales vary; 160 pixels is not a common angular footprint.

The exact field-qualified ID population was verified against the original
FITS catalogues and extraction coordinates. Its SHA256 is
`1b8a58e05dde752307228b62a352a2366bb63309452c2da1f0bc419375b42f57`,
computed from UTF-8 newline-joined lexicographically sorted `object_id` values
without a trailing newline. The release preparer independently compares final
HATS IDs, fields and coordinates with that extraction cache.

## Schema and units

| Column | Meaning |
| --- | --- |
| `object_id` | Field-qualified identifier, e.g. `GOODS-S:<ID>` |
| `field` | `GOODS-S` or `GOODS-N` |
| `ra`, `dec` | ICRS degrees |
| `image.band` | Ordered list of available channel names; determines array channel order |
| `image.flux` | Per-channel 160 × 160 arrays, nJy per pixel |
| `image.ivar` | Inverse variance consistent with flux units |
| `image.mask` | Boolean validity arrays; **True means valid** |
| `image.scale` | Per-channel pixel scale, arcseconds per pixel |
| `image.psf_fwhm` | Per-channel PSF FWHM metadata in arcseconds; not PSF image stamps |
| `image_sources` | Provenance of selected upstream mosaics |
| `phot_<band>_circ3_flux_njy`, `..._err_njy` | Circular-aperture photometry and uncertainty |
| `phot_<band>_kron_flux_njy`, `..._err_njy` | Kron photometry and uncertainty |
| `phot_<band>_kron_conv_flux_njy`, `..._err_njy` | Available NIRCam convolved Kron measurements |
| `photoz`, `photoz_kron` | EAZY redshift estimates, intervals and posterior information |
| `z_spec` | Spectroscopic redshift where available; nullable |
| `size_a`, `size_b` | Upstream morphology values; consult the DR5 column definitions for units |

Missing scalar measurements may be null. Invalid image pixels are zero-filled;
use the mask, since zero alone does not indicate validity. Consult the complete
Arrow schema in `release/provenance.json`; do not assume every source has all
potential filters, that zero-filled pixels are measurements, or that all sources
have spectroscopic labels. No spectra or HST image cutouts are included here.

## Using the collection

After downloading the complete repository to a local directory:

```python
import hats
import lsdb

root = "/path/to/jades-jwst"
collection = hats.read_hats(root)
catalog = lsdb.open_catalog(root + "/jades_jwst")
print(catalog.schema)
# Select columns and a bounded sky region before computing image data.
```

Use HATS-aware spatial readers. Recursively loading every Parquet file with
`datasets.load_dataset("parquet", ...)` would mix main and margin data and can
duplicate sources. Keep the margin for spatial joins. The original build used
Python 3.11.11, hats/hats-import 0.9.0, pyarrow 23.0.1 and dask/distributed
2025.3.1. Pin the Hugging Face repository revision in training configurations.

## Validation and limitations

Release checks cover collection relocation/readability, 527 main partitions,
484,219 rows, unique IDs, field counts, finite coordinates, identical partition
schemas and exact equality of final IDs/fields/coordinates to the preflight
cache. They read scalar columns only. They do not constitute a full pixel-level
audit or prove full-band coverage. Input selection is inherited from the DR5
catalogues and image coverage, and the two small deep fields are not a
representative all-sky sample.

For AION-2, define sky/source-aware splits and account for overlapping cutouts
and other surveys on the same sky. Avoid leakage through repeated sources,
margin copies or overlapping train/evaluation images. Validate band mapping,
pixel scales, masks and nullable labels in a real training-loader batch before
training; no benchmark or performance claim accompanies this export.

## Reproducibility and provenance

The successful extraction/HATS implementation is pinned to
[MultimodalUniverse commit 6c6b7ab](https://github.com/MaxRonce/MultimodalUniverse/tree/6c6b7aba1d3ec5e741af69fb58da8460dbb247d8).
See the [builder](https://github.com/MaxRonce/MultimodalUniverse/blob/6c6b7aba1d3ec5e741af69fb58da8460dbb247d8/scripts/jades/build_parent_sample_hats.py),
[manifest generator](https://github.com/MaxRonce/MultimodalUniverse/blob/6c6b7aba1d3ec5e741af69fb58da8460dbb247d8/scripts/jades/make_manifest.py),
[HATS importer](https://github.com/MaxRonce/MultimodalUniverse/blob/6c6b7aba1d3ec5e741af69fb58da8460dbb247d8/mmu/hats_import.py)
and [Rusty build runbook](https://github.com/MaxRonce/MultimodalUniverse/blob/6c6b7aba1d3ec5e741af69fb58da8460dbb247d8/scripts/jades/RUSTY_RUNBOOK.md).
The successful run used order 12, pixel threshold 2048, minimum order 4,
import batch size 4 and 8 Dask workers. The source manifest is bundled in
`release/input_manifest.json`; filenames alone are not cryptographic FITS checksums.
The packaging script and its hash are also bundled, alongside the publication guide.

## Licensing and attribution

The [MAST JADES HLSP page](https://archive.stsci.edu/hlsp/jades#citations)
states CC BY 4.0 for its HLSP data products. The exact inputs here are DR5
products from the [official DR5 portal](https://slate.ucsc.edu/~brant/jades-dr5/)
and [MIRI v1.0 release](https://github.com/staceyalberts/JWST-JADES-MIRI-PARALLELS/releases/tag/v1.0).
The MAST statement has not yet been shown to cover these exact DR5 files.
Record DR5-specific licence evidence in `release/` before public redistribution.
Only then set the card's licence metadata, include the applicable terms and
replace this pending-status notice. The pipeline's MIT code licence does not
relicense the observational data. MMU's dataset policy does not replace upstream terms.

Cite the original survey and products, as well as this derived release and its
code revision. The DR5 portal identifies Johnson et al. (2026,
[arXiv:2601.15954](https://arxiv.org/abs/2601.15954)) for NIRCam,
Robertson et al. (2026, [arXiv:2601.15956](https://arxiv.org/abs/2601.15956))
for photometric catalogues and Alberts et al. (2026,
[arXiv:2601.15955](https://arxiv.org/abs/2601.15955)) for MIRI.
Follow the portal's complete bibliography and programme acknowledgements,
including the JADES overview and NASA/ESA/CSA JWST, MAST and STScI attribution
requirements. This derived dataset is a reformatting and selection of upstream
products, not an official JADES release. No DOI for this derived release has
been assigned in this preparation.
