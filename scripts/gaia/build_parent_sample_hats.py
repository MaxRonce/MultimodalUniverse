"""Convert raw Gaia DR3 bulk download files into a HATS catalog.

Matches v1 MMU's ``scripts/gaia/gaia.py`` HuggingFace schema 1:1, joining the
raw Gaia Archive bulk tables on ``source_id`` directly (duplicating what v1's
``merge_parts.py`` does) instead of consuming a pre-built intermediate.

Cluster layout::

    /mnt/ceph/users/polymathic/external_data/astro/Gaia/
        GaiaSource_{start}-{end}.hdf5              # main DR3 source catalog
        XpContinuousMeanSpectrum_{start}-{end}.hdf5 # BP/RP coefficients
        AstrophysicalParameters_{start}-{end}.hdf5  # (not used — all needed
                                                     # gspphot fields are
                                                     # mirrored in GaiaSource)
        RvsMeanSpectrum_{start}-{end}.hdf5          # (not used — v1 only
                                                     # exposes the scalar RV
                                                     # fields from GaiaSource)

The GaiaSource and XpContinuousMeanSpectrum shards share the same ``{start}-{end}``
suffix scheme (the "parts" numbering from the Gaia Archive's bulk split). For
every part, every XP source_id is a subset of the corresponding GaiaSource
source_id set — v1 ``merge_parts.py`` asserts this.

This script, for each part:

1. Opens the XP file and loads source_ids + bp/rp coefficients. This is the
   smaller of the two (~16% of GaiaSource rows). It's the defining set —
   only sources with XP spectra end up in the Gaia HATS catalog.
2. Opens the GaiaSource file and loads source_ids + all required columns.
3. Inner-joins on source_id (np.intersect1d, matching v1's approach).
4. Applies the cone cut (if one is active) on the joined ra/dec BEFORE
   building the struct columns.
5. Builds a PyArrow table where each row has:
     ``spectral_coefficients`` (struct<coeff: list<f32>, coeff_error: list<f32>>),
     ``photometry``            (struct<13 scalar fields>),
     ``astrometry``            (struct<20 scalar fields>),
     ``radial_velocity``       (struct<5 scalar fields>),
     ``gspphot``               (struct<21 scalar fields>),
     ``flags``                 (struct<1 scalar field: ruwe>),
     ``corrections``           (struct<7 scalar fields>),
     ``object_id`` (int64, source_id),
     ``ra``, ``dec`` (float64, duplicated from astrometry for HATS spatial indexing)

The struct-of-parallel-scalar-fields layout survives hats-import's
``nested_pandas`` round-trip (no extension types anywhere). See
``project_image_storage`` memory for details on the nested_pandas pitfalls.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
import pyarrow as pa

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import write_hats


CATALOG_NAME = "gaia"


# Field lists copied verbatim from v1 scripts/gaia/gaia.py. These must stay
# in sync with v1; if v1 ever adds/removes a column, this script must too.
SPECTRUM_FEATURES = ["coeff", "coeff_error"]

PHOTOMETRY_FEATURES = [
    "phot_g_mean_mag", "phot_g_mean_flux", "phot_g_mean_flux_error",
    "phot_bp_mean_mag", "phot_bp_mean_flux", "phot_bp_mean_flux_error",
    "phot_rp_mean_mag", "phot_rp_mean_flux", "phot_rp_mean_flux_error",
    "phot_bp_rp_excess_factor", "bp_rp", "bp_g", "g_rp",
]

ASTROMETRY_FEATURES = [
    "ra", "ra_error", "dec", "dec_error",
    "parallax", "parallax_error",
    "pmra", "pmra_error", "pmdec", "pmdec_error",
    "ra_dec_corr", "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr",
    "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr",
    "parallax_pmra_corr", "parallax_pmdec_corr", "pmra_pmdec_corr",
]

RV_FEATURES = [
    "radial_velocity", "radial_velocity_error",
    "rv_template_fe_h", "rv_template_logg", "rv_template_teff",
]

GSPPHOT_FEATURES = [
    "ag_gspphot", "ag_gspphot_lower", "ag_gspphot_upper",
    "azero_gspphot", "azero_gspphot_lower", "azero_gspphot_upper",
    "distance_gspphot", "distance_gspphot_lower", "distance_gspphot_upper",
    "ebpminrp_gspphot", "ebpminrp_gspphot_lower", "ebpminrp_gspphot_upper",
    "logg_gspphot", "logg_gspphot_lower", "logg_gspphot_upper",
    "mh_gspphot", "mh_gspphot_lower", "mh_gspphot_upper",
    "teff_gspphot", "teff_gspphot_lower", "teff_gspphot_upper",
]

FLAG_FEATURES = ["ruwe"]

CORRECTION_FEATURES = [
    "ecl_lat", "ecl_lon",
    "nu_eff_used_in_astrometry", "pseudocolour",
    "astrometric_params_solved",
    "rv_template_teff",  # v1 has this duplicated in both RV and corrections
    "grvs_mag",
]

ALL_SOURCE_COLUMNS = (
    ["source_id", "ra", "dec"]
    + PHOTOMETRY_FEATURES
    + ASTROMETRY_FEATURES
    + RV_FEATURES
    + GSPPHOT_FEATURES
    + FLAG_FEATURES
    + CORRECTION_FEATURES
)
# ra and dec appear in both ASTROMETRY_FEATURES and the top-level ra/dec, and
# are only read once from the HDF5 file; dedupe to avoid h5py errors.
ALL_SOURCE_COLUMNS = list(dict.fromkeys(ALL_SOURCE_COLUMNS))


PART_RE = re.compile(r"_(\d+)-(\d+)\.hdf5$")


def find_shard_pairs(raw_root: str) -> list[tuple[str, str]]:
    """Pair up GaiaSource_*.hdf5 with XpContinuousMeanSpectrum_*.hdf5 shards
    by their ``{start}-{end}`` suffix.

    Returns a list of ``(source_path, xp_path)`` tuples in sorted order.
    """
    sources = {}
    xps = {}
    for p in glob.glob(os.path.join(raw_root, "GaiaSource_*.hdf5")):
        m = PART_RE.search(p)
        if m:
            sources[m.group(0)] = p
    for p in glob.glob(os.path.join(raw_root, "XpContinuousMeanSpectrum_*.hdf5")):
        m = PART_RE.search(p)
        if m:
            xps[m.group(0)] = p
    common = sorted(set(sources) & set(xps))
    return [(sources[k], xps[k]) for k in common]


def _read_xp_shard(xp_path: str) -> dict[str, np.ndarray]:
    """Load the XP source_ids and coefficient arrays from one shard.

    Returns a dict with:
        source_id: (N,) int64
        coeff:     (N, 110) float32  (55 bp + 55 rp)
        coeff_error: (N, 110) float32
    """
    with h5py.File(xp_path, "r") as f:
        source_id = np.asarray(f["source_id"][:], dtype=np.int64)
        bp_c = np.asarray(f["bp_coefficients"][:], dtype=np.float32)
        rp_c = np.asarray(f["rp_coefficients"][:], dtype=np.float32)
        bp_e = np.asarray(f["bp_coefficient_errors"][:], dtype=np.float32)
        rp_e = np.asarray(f["rp_coefficient_errors"][:], dtype=np.float32)
    coeff = np.concatenate([bp_c, rp_c], axis=-1).astype(np.float32)
    coeff_error = np.concatenate([bp_e, rp_e], axis=-1).astype(np.float32)
    return {"source_id": source_id, "coeff": coeff, "coeff_error": coeff_error}


def _read_source_columns(source_path: str, columns: list[str]) -> dict[str, np.ndarray]:
    """Load ``columns`` from a GaiaSource shard.

    Any column missing from the file is returned as an array of NaN (for float
    cols) or zeros (for int cols) so that v1's schema stays uniform across
    Gaia DR3 data releases where column availability can vary slightly.
    """
    with h5py.File(source_path, "r") as f:
        n = f["source_id"].shape[0]
        out: dict[str, np.ndarray] = {}
        for c in columns:
            if c in f:
                out[c] = np.asarray(f[c][:])
            else:
                out[c] = np.full(n, np.nan, dtype=np.float32)
    return out


def process_shard(
    source_path: str,
    xp_path: str,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> pa.Table | None:
    """Process one (source, xp) shard pair into a PyArrow table.

    Returns ``None`` if the cone cut leaves zero rows.
    """
    xp = _read_xp_shard(xp_path)
    src = _read_source_columns(source_path, ALL_SOURCE_COLUMNS)

    # Inner-join on source_id. Both sides are sorted by source_id within the
    # shard, but we use np.intersect1d with return_indices to be safe.
    _, ix_xp, ix_src = np.intersect1d(
        xp["source_id"], src["source_id"], return_indices=True, assume_unique=True
    )
    n = ix_xp.size
    if n == 0:
        return None

    # Gather the joined rows from both sides.
    coeff = xp["coeff"][ix_xp]
    coeff_error = xp["coeff_error"][ix_xp]
    source_id = xp["source_id"][ix_xp]
    src_joined = {c: src[c][ix_src] for c in ALL_SOURCE_COLUMNS if c != "source_id"}

    ra = np.asarray(src_joined["ra"], dtype=np.float64)
    dec = np.asarray(src_joined["dec"], dtype=np.float64)

    if ra_center is not None and dec_center is not None and radius is not None:
        mask = apply_cone_filter(ra, dec, ra_center, dec_center, radius)
        if not mask.any():
            return None
        coeff = coeff[mask]
        coeff_error = coeff_error[mask]
        source_id = source_id[mask]
        ra = ra[mask]
        dec = dec[mask]
        src_joined = {k: v[mask] for k, v in src_joined.items()}
        n = mask.sum()

    def _struct(feature_list: list[str]) -> pa.StructArray:
        """Build a struct array from a list of float32 scalar columns."""
        arrays = [
            pa.array(np.asarray(src_joined[f], dtype=np.float32), type=pa.float32())
            for f in feature_list
        ]
        return pa.StructArray.from_arrays(arrays, names=feature_list)

    spectral_struct = pa.StructArray.from_arrays(
        [
            pa.array(
                [list(row) for row in coeff],
                type=pa.list_(pa.float32()),
            ),
            pa.array(
                [list(row) for row in coeff_error],
                type=pa.list_(pa.float32()),
            ),
        ],
        names=["coeff", "coeff_error"],
    )

    columns: dict[str, pa.Array] = {
        "ra": pa.array(ra, type=pa.float64()),
        "dec": pa.array(dec, type=pa.float64()),
        "object_id": pa.array(source_id.astype(np.int64), type=pa.int64()),
        "spectral_coefficients": spectral_struct,
        "photometry": _struct(PHOTOMETRY_FEATURES),
        "astrometry": _struct(ASTROMETRY_FEATURES),
        "radial_velocity": _struct(RV_FEATURES),
        "gspphot": _struct(GSPPHOT_FEATURES),
        "flags": _struct(FLAG_FEATURES),
        "corrections": _struct(CORRECTION_FEATURES),
    }
    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap on number of (source, xp) shard pairs to process.")
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone radius in degrees; requires --ra-center/--dec-center.")
    args = parser.parse_args(argv)

    pairs = find_shard_pairs(args.raw_root)
    if args.max_files is not None:
        pairs = pairs[:args.max_files]
    if not pairs:
        print(f"ERROR: no GaiaSource/XP shard pairs under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Found {len(pairs)} (source, xp) shard pair(s)")

    tables: list[pa.Table] = []
    total = 0
    for i, (source_path, xp_path) in enumerate(pairs, 1):
        t = process_shard(
            source_path, xp_path,
            ra_center=args.ra_center,
            dec_center=args.dec_center,
            radius=args.radius,
        )
        if t is None:
            continue
        tables.append(t)
        total += t.num_rows
        print(f"  [{i}/{len(pairs)}] {os.path.basename(xp_path)}: {t.num_rows} rows")

    if not tables:
        print("ERROR: no rows survived the cone cut", file=sys.stderr)
        return 1

    print(f"\nWriting HATS catalog: {total} rows from {len(tables)} shard(s)")
    catalog_dir = write_hats(
        tables,
        output_path=args.output_root,
        catalog_name=CATALOG_NAME,
        pixel_threshold=args.pixel_threshold,
    )
    print(f"Done: {catalog_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
