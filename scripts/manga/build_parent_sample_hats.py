"""Convert raw SDSS-IV MaNGA into a HATS catalog.

Matches v1 MMU's ``scripts/manga/build_parent_sample.py`` exactly, with
HATS output instead of HDF5:

    1. Read drpall-v3_1_1.fits and dapall-v3_1_1-3.1.0.fits, inner-join on
       plateifu, keep only DAPDONE rows.
    2. (If a cone cut is active) trim the joined catalog by ifura/ifudec.
    3. For each surviving plate-ifu:
         - Open ``redux/v3_1_1/{plate}/stack/manga-{plateifu}-LOGCUBE.fits.gz``
           and extract flux/ivar/mask/lsf/wave + griz reconstructed images
           and PSFs. Pad spatial dims to 96×96.
         - Open ``analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/{plate}/{ifu}/
           manga-{plateifu}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz`` for the DAP
           analysis maps + spaxel coordinate grids. Pad to 96×96.
    4. Build a per-row dict matching v1's HuggingFace `Features(...)`:

           spaxels: struct<flux, ivar, mask, lsf, lambda, x, y, spaxel_idx,
                            *_units, skycoo_x/y, ellcoo_r/rre/rkpc/theta, *_units>
           images:  struct<filter, flux, flux_units, psf, psf_units, scale, scale_units>
           maps:    struct<group, label, flux, ivar, mask, array_units>
           ra, dec, object_id (= plateifu), z, spaxel_size, spaxel_size_units

       All arrays stored as plain nested lists (no extension types) — see
       ``project_image_storage`` memory.
    5. Write HATS via ``mmu.hats_import.write_hats``.

Cluster layout::

    /mnt/ceph/users/polymathic/external_data/astro/manga/
        drpall-v3_1_1.fits                                  # plate-ifu summary catalog
        dapall-v3_1_1-3.1.0.fits                            # DAP done flag, etc.
        dr17/manga/spectro/redux/v3_1_1/{plate}/stack/manga-{plateifu}-LOGCUBE.fits.gz
        dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/{plate}/{ifu}/
            manga-{plateifu}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
from astropy.io import fits
from astropy.table import Table, join

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import np_to_pyarrow_list, write_hats


CATALOG_NAME = "manga"

IMAGE_SIZE = 96
SPECTRUM_SIZE = 4563
N_BANDS = 4
BANDS = ["g", "r", "i", "z"]
SPAXEL_SIZE_ARCSEC = 0.5
DAPTYPE = "HYB10-MILESHC-MASTARSSP"


def _b2s(v) -> str:
    return v.decode().strip() if isinstance(v, (bytes, np.bytes_)) else str(v).strip()


def load_catalog(raw_root: str) -> Table:
    """Read drpall + dapall and inner-join on plate-ifu, keeping DAPDONE rows."""
    drpall = Table.read(os.path.join(raw_root, "drpall-v3_1_1.fits"), hdu="MANGA")
    dapall = Table.read(
        os.path.join(raw_root, "dapall-v3_1_1-3.1.0.fits"),
        hdu=DAPTYPE,
    )
    catalog = join(
        drpall, dapall,
        keys_left="plateifu", keys_right="PLATEIFU", join_type="inner",
    )
    catalog = catalog[np.asarray(catalog["DAPDONE"]).astype(bool)]
    return catalog


def cube_path(raw_root: str, plateifu: str) -> str:
    plate, _ = plateifu.split("-")
    return os.path.join(
        raw_root, "dr17", "manga", "spectro", "redux", "v3_1_1",
        plate, "stack", f"manga-{plateifu}-LOGCUBE.fits.gz",
    )


def maps_path(raw_root: str, plateifu: str) -> str:
    plate, ifu = plateifu.split("-")
    return os.path.join(
        raw_root, "dr17", "manga", "spectro", "analysis", "v3_1_1", "3.1.0",
        DAPTYPE, plate, ifu,
        f"manga-{plateifu}-MAPS-{DAPTYPE}.fits.gz",
    )


def _pad_spatial(arr: np.ndarray) -> np.ndarray:
    """Pad the LAST two axes of ``arr`` symmetrically up to ``IMAGE_SIZE``."""
    *_, ny, nx = arr.shape
    pad_y = (IMAGE_SIZE - ny) // 2
    pad_x = (IMAGE_SIZE - nx) // 2
    if pad_y == 0 and pad_x == 0:
        return arr
    pads = [(0, 0)] * (arr.ndim - 2) + [
        (pad_y, IMAGE_SIZE - ny - pad_y),
        (pad_x, IMAGE_SIZE - nx - pad_x),
    ]
    return np.pad(arr, pads)


def _to_native(arr: np.ndarray) -> np.ndarray:
    """Make the buffer little-endian native so PyArrow accepts it."""
    if arr.dtype.byteorder == ">":
        return arr.byteswap().view(arr.dtype.newbyteorder("<"))
    return arr


def process_cube(
    plateifu: str,
    cube_file: str,
    map_file: str,
) -> dict | None:
    """Read one MaNGA plate-ifu (LOGCUBE + DAP MAPS) into a per-row dict.

    Returns ``None`` if either input file is missing.
    """
    if not os.path.exists(cube_file):
        return None
    if not os.path.exists(map_file):
        return None

    with fits.open(cube_file) as cube:
        flux = _to_native(np.asarray(cube["FLUX"].data, dtype=np.float32))
        ivar = _to_native(np.asarray(cube["IVAR"].data, dtype=np.float32))
        mask = _to_native(np.asarray(cube["MASK"].data, dtype=np.int64))
        lsf = _to_native(np.asarray(cube["LSFPOST"].data, dtype=np.float32))
        wave = _to_native(np.asarray(cube["WAVE"].data, dtype=np.float32))
        flux_units = cube["FLUX"].header.get("BUNIT", "")
        lambda_units = cube["FLUX"].header.get("CUNIT3", "")

        # Pad spatial axes (last two) to IMAGE_SIZE; mask gets DONOTUSE pad value.
        flux = _pad_spatial(flux)
        ivar = _pad_spatial(ivar)
        mask = np.pad(
            mask,
            [(0, 0)] + [
                ((IMAGE_SIZE - mask.shape[1]) // 2, IMAGE_SIZE - mask.shape[1] - (IMAGE_SIZE - mask.shape[1]) // 2),
                ((IMAGE_SIZE - mask.shape[2]) // 2, IMAGE_SIZE - mask.shape[2] - (IMAGE_SIZE - mask.shape[2]) // 2),
            ],
            constant_values=1024,
        )
        lsf = _pad_spatial(lsf)

        nwave = flux.shape[0]
        nspaxels = IMAGE_SIZE * IMAGE_SIZE  # 9216

        # Reshape per-spaxel spectra to (nspaxels, nwave) — one row per spaxel.
        flux_2d = flux.reshape(nwave, nspaxels).T  # (nspaxels, nwave)
        ivar_2d = ivar.reshape(nwave, nspaxels).T
        mask_2d = mask.reshape(nwave, nspaxels).T
        lsf_2d = lsf.reshape(nwave, nspaxels).T
        # Wavelength is shared by all spaxels but v1 stores it per-spaxel; we
        # repeat to keep per-spaxel symmetry.
        lam_2d = np.tile(wave[None, :], (nspaxels, 1))

        # Spaxel x/y indices and unique idx.
        yy, xx = np.indices((IMAGE_SIZE, IMAGE_SIZE))
        x_arr = xx.reshape(nspaxels).astype(np.int8)
        y_arr = yy.reshape(nspaxels).astype(np.int8)
        spaxel_idx = np.arange(nspaxels, dtype=np.int16)

        # Reconstructed griz images and PSFs.
        img_stack = np.stack([
            _pad_spatial(_to_native(np.asarray(cube[f"{b.upper()}IMG"].data, dtype=np.float32)))
            for b in BANDS
        ])
        psf_stack = np.stack([
            _pad_spatial(_to_native(np.asarray(cube[f"{b.upper()}PSF"].data, dtype=np.float32)))
            for b in BANDS
        ])

    # DAP MAPS file: spaxel coordinate grids + analysis maps.
    with fits.open(map_file) as mapf:
        skycoo = _to_native(np.asarray(mapf["SPX_SKYCOO"].data, dtype=np.float32))
        skycoo = _pad_spatial(skycoo)  # shape (2, 96, 96)
        skycoo_units = mapf["SPX_SKYCOO"].header.get("BUNIT", "")

        ellcoo = _to_native(np.asarray(mapf["SPX_ELLCOO"].data, dtype=np.float32))
        ellcoo = _pad_spatial(ellcoo)  # shape (4, 96, 96): r, r/re, r_kpc, theta
        ellcoo_units = [
            mapf["SPX_ELLCOO"].header.get(f"U{i}", "")
            for i in range(1, 5)
        ]

        # Walk every non-PRIMARY, non-IVAR/MASK extension as a "map".
        maps_data: list[dict] = []
        for ext in mapf:
            if ext.name == "PRIMARY":
                continue
            if ext.name.endswith(("IVAR", "MASK")):
                continue
            arr = ext.data
            if arr is None:
                continue
            arr = _to_native(np.asarray(arr, dtype=np.float32))
            arr = _pad_spatial(arr)
            errdata = ext.header.get("ERRDATA")
            qualdata = ext.header.get("QUALDATA")
            if errdata and errdata in mapf:
                err = _to_native(np.asarray(mapf[errdata].data, dtype=np.float32))
                err = _pad_spatial(err)
            else:
                err = np.zeros_like(arr)
            if qualdata and qualdata in mapf:
                qual = _to_native(np.asarray(mapf[qualdata].data, dtype=np.float32))
                qual = _pad_spatial(qual)
            else:
                qual = np.full_like(arr, 1073741824.0)
            unit = ext.header.get("BUNIT", "")
            base_name = ext.name.lower()
            if arr.ndim == 3:
                # Multi-channel: emit one map per channel.
                for ch in range(arr.shape[0]):
                    chan_label = (
                        ext.header.get(f"C{ch + 1:02}", ext.header.get(f"C{ch + 1}", ""))
                    ).replace("-", "_").strip().replace(". ", "").replace(" ", "_")
                    chan_unit = (
                        ext.header.get(f"U{ch + 1:02}", ext.header.get(f"U{ch + 1}", ""))
                    ) or unit
                    maps_data.append({
                        "group": base_name,
                        "label": f"{base_name}_{chan_label.lower()}",
                        "flux": arr[ch],
                        "ivar": err[ch] if err.ndim == 3 else err,
                        "mask": qual[ch] if qual.ndim == 3 else qual,
                        "array_units": chan_unit,
                    })
            else:
                maps_data.append({
                    "group": base_name,
                    "label": base_name,
                    "flux": arr,
                    "ivar": err,
                    "mask": qual,
                    "array_units": unit,
                })

    return {
        "plateifu": plateifu,
        "spaxels": {
            "flux": flux_2d, "ivar": ivar_2d, "mask": mask_2d,
            "lsf": lsf_2d, "lambda": lam_2d,
            "x": x_arr, "y": y_arr, "spaxel_idx": spaxel_idx,
            "flux_units": [flux_units] * nspaxels,
            "lambda_units": [lambda_units] * nspaxels,
            "skycoo_x": skycoo[0].reshape(nspaxels),
            "skycoo_y": skycoo[1].reshape(nspaxels),
            "ellcoo_r": ellcoo[0].reshape(nspaxels),
            "ellcoo_rre": ellcoo[1].reshape(nspaxels),
            "ellcoo_rkpc": ellcoo[2].reshape(nspaxels),
            "ellcoo_theta": ellcoo[3].reshape(nspaxels),
            "skycoo_units": [skycoo_units] * nspaxels,
            "ellcoo_r_units": [ellcoo_units[0]] * nspaxels,
            "ellcoo_rre_units": [ellcoo_units[1]] * nspaxels,
            "ellcoo_rkpc_units": [ellcoo_units[2]] * nspaxels,
            "ellcoo_theta_units": [ellcoo_units[3]] * nspaxels,
        },
        "images": {
            "filter": list(BANDS),
            "flux": img_stack,                  # (4, 96, 96)
            "flux_units": ["nanomaggies/pixel"] * N_BANDS,
            "psf": psf_stack,
            "psf_units": ["nanomaggies/pixel"] * N_BANDS,
            "scale": [SPAXEL_SIZE_ARCSEC] * N_BANDS,
            "scale_units": ["arcsec"] * N_BANDS,
        },
        "maps": maps_data,
    }


def _build_spaxels_struct(records: list[dict]) -> pa.StructArray:
    """Convert per-row spaxel dicts into a single struct column.

    The 2D float arrays use the fast ``np_to_pyarrow_list`` path; everything
    else is built directly with pa.array.
    """
    nrows = len(records)
    spx_list = [r["spaxels"] for r in records]
    nspax_per_row = [len(s["x"]) for s in spx_list]

    def _2d_list_col(name: str, dtype) -> pa.Array:
        """Build a list<list<dtype>> column where row i is the per-spaxel
        flux array for object i (shape ``(nspaxels_i, nwave)``)."""
        # Concatenate all rows' (nspaxels, nwave) → (sum_nspaxels, nwave)
        # then build the inner list<float32> via offset arithmetic, then
        # group by row using outer offsets.
        arrays = [s[name].astype(dtype) for s in spx_list]
        if not arrays:
            return pa.array([], type=pa.list_(pa.list_(pa.from_numpy_dtype(dtype))))
        all_concat = np.concatenate(arrays, axis=0)
        # inner list (per-spaxel spectrum)
        inner = np_to_pyarrow_list(all_concat)
        # outer list: offsets in units of inner-list length
        outer_offsets = np.zeros(nrows + 1, dtype=np.int32)
        for i, n in enumerate(nspax_per_row):
            outer_offsets[i + 1] = outer_offsets[i] + n
        return pa.ListArray.from_arrays(values=inner, offsets=outer_offsets)

    def _flat_list_col(name: str, dtype, pa_type) -> pa.Array:
        """Build a list<dtype> column where row i is a flat per-spaxel array."""
        arrays = [np.asarray(s[name], dtype=dtype) for s in spx_list]
        offsets = np.zeros(nrows + 1, dtype=np.int32)
        for i, a in enumerate(arrays):
            offsets[i + 1] = offsets[i] + len(a)
        if arrays:
            values = pa.array(np.concatenate(arrays), type=pa_type)
        else:
            values = pa.array([], type=pa_type)
        return pa.ListArray.from_arrays(values=values, offsets=offsets)

    def _str_list_col(name: str) -> pa.Array:
        offsets = np.zeros(nrows + 1, dtype=np.int32)
        flat: list[str] = []
        for i, s in enumerate(spx_list):
            flat.extend(s[name])
            offsets[i + 1] = offsets[i] + len(s[name])
        return pa.ListArray.from_arrays(
            values=pa.array(flat, type=pa.string()),
            offsets=offsets,
        )

    fields = {
        "flux":      _2d_list_col("flux", np.float32),
        "ivar":      _2d_list_col("ivar", np.float32),
        "mask":      _2d_list_col("mask", np.int64),
        "lsf":       _2d_list_col("lsf", np.float32),
        "lambda":    _2d_list_col("lambda", np.float32),
        "x":         _flat_list_col("x", np.int8, pa.int8()),
        "y":         _flat_list_col("y", np.int8, pa.int8()),
        "spaxel_idx": _flat_list_col("spaxel_idx", np.int16, pa.int16()),
        "flux_units":   _str_list_col("flux_units"),
        "lambda_units": _str_list_col("lambda_units"),
        "skycoo_x":    _flat_list_col("skycoo_x", np.float32, pa.float32()),
        "skycoo_y":    _flat_list_col("skycoo_y", np.float32, pa.float32()),
        "ellcoo_r":    _flat_list_col("ellcoo_r", np.float32, pa.float32()),
        "ellcoo_rre":  _flat_list_col("ellcoo_rre", np.float32, pa.float32()),
        "ellcoo_rkpc": _flat_list_col("ellcoo_rkpc", np.float32, pa.float32()),
        "ellcoo_theta": _flat_list_col("ellcoo_theta", np.float32, pa.float32()),
        "skycoo_units":      _str_list_col("skycoo_units"),
        "ellcoo_r_units":    _str_list_col("ellcoo_r_units"),
        "ellcoo_rre_units":  _str_list_col("ellcoo_rre_units"),
        "ellcoo_rkpc_units": _str_list_col("ellcoo_rkpc_units"),
        "ellcoo_theta_units": _str_list_col("ellcoo_theta_units"),
    }
    return pa.StructArray.from_arrays(list(fields.values()), names=list(fields.keys()))


def _build_images_struct(records: list[dict]) -> pa.StructArray:
    n = len(records)
    bands = pa.array([r["images"]["filter"] for r in records], type=pa.list_(pa.string()))

    def _img_col(key: str) -> pa.Array:
        # records[i]["images"][key] has shape (4, 96, 96).
        nested = [
            [[list(row) for row in band] for band in r["images"][key]]
            for r in records
        ]
        return pa.array(nested, type=pa.list_(pa.list_(pa.list_(pa.float32()))))

    return pa.StructArray.from_arrays(
        [
            bands,
            _img_col("flux"),
            pa.array([r["images"]["flux_units"] for r in records], type=pa.list_(pa.string())),
            _img_col("psf"),
            pa.array([r["images"]["psf_units"] for r in records], type=pa.list_(pa.string())),
            pa.array([r["images"]["scale"] for r in records], type=pa.list_(pa.float32())),
            pa.array([r["images"]["scale_units"] for r in records], type=pa.list_(pa.string())),
        ],
        names=["filter", "flux", "flux_units", "psf", "psf_units", "scale", "scale_units"],
    )


def _build_maps_struct(records: list[dict]) -> pa.StructArray:
    n = len(records)
    groups = []
    labels = []
    flux_per_row = []
    ivar_per_row = []
    mask_per_row = []
    units = []
    for r in records:
        gs, ls, fl, iv, mk, un = [], [], [], [], [], []
        for m in r["maps"]:
            gs.append(m["group"])
            ls.append(m["label"])
            fl.append([list(row) for row in m["flux"]])
            iv.append([list(row) for row in m["ivar"]])
            mk.append([list(row) for row in m["mask"]])
            un.append(m["array_units"])
        groups.append(gs)
        labels.append(ls)
        flux_per_row.append(fl)
        ivar_per_row.append(iv)
        mask_per_row.append(mk)
        units.append(un)

    return pa.StructArray.from_arrays(
        [
            pa.array(groups, type=pa.list_(pa.string())),
            pa.array(labels, type=pa.list_(pa.string())),
            pa.array(flux_per_row, type=pa.list_(pa.list_(pa.list_(pa.float32())))),
            pa.array(ivar_per_row, type=pa.list_(pa.list_(pa.list_(pa.float32())))),
            pa.array(mask_per_row, type=pa.list_(pa.list_(pa.list_(pa.float32())))),
            pa.array(units, type=pa.list_(pa.string())),
        ],
        names=["group", "label", "flux", "ivar", "mask", "array_units"],
    )


def build_table(records: list[dict], catalog: Table) -> pa.Table:
    """Stitch the per-cube records back to the joined catalog and build the
    PyArrow table."""
    # Build a plateifu → catalog row map for the surviving plate-ifus.
    cat_lookup = {}
    plateifu_col = np.array([_b2s(p) for p in catalog["plateifu"]])
    for i, p in enumerate(plateifu_col):
        cat_lookup[p] = i

    keep_records: list[dict] = []
    keep_indices: list[int] = []
    for r in records:
        i = cat_lookup.get(_b2s(r["plateifu"]))
        if i is None:
            continue
        keep_records.append(r)
        keep_indices.append(i)
    if not keep_records:
        raise RuntimeError("No records joined to catalog rows")

    catalog_subset = catalog[np.array(keep_indices)]
    ra = np.asarray(catalog_subset["ifura"], dtype=np.float64)
    dec = np.asarray(catalog_subset["ifudec"], dtype=np.float64)
    z = np.asarray(catalog_subset["nsa_z"], dtype=np.float32)
    plateifu_arr = np.array([_b2s(p) for p in catalog_subset["plateifu"]])

    columns: dict[str, pa.Array] = {
        "ra": pa.array(ra),
        "dec": pa.array(dec),
        "object_id": pa.array(plateifu_arr, type=pa.string()),
        "z": pa.array(z, type=pa.float32()),
        "spaxel_size": pa.array(
            np.full(len(keep_records), SPAXEL_SIZE_ARCSEC, dtype=np.float32),
            type=pa.float32(),
        ),
        "spaxel_size_units": pa.array(
            ["arcsec"] * len(keep_records), type=pa.string()
        ),
        "spaxels": _build_spaxels_struct(keep_records),
        "images":  _build_images_struct(keep_records),
        "maps":    _build_maps_struct(keep_records),
    }
    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap on number of plate-ifus to process.")
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone radius in degrees; requires --ra-center/--dec-center.")
    args = parser.parse_args(argv)

    print(f"Loading drpall + dapall from {args.raw_root}")
    catalog = load_catalog(args.raw_root)
    print(f"  {len(catalog)} plate-ifus after DAPDONE join")

    if args.ra_center is not None and args.dec_center is not None and args.radius is not None:
        mask = apply_cone_filter(
            np.asarray(catalog["ifura"], dtype=np.float64),
            np.asarray(catalog["ifudec"], dtype=np.float64),
            args.ra_center, args.dec_center, args.radius,
        )
        catalog = catalog[mask]
        print(
            f"  {len(catalog)} plate-ifus after cone cut "
            f"(ra={args.ra_center}, dec={args.dec_center}, radius={args.radius})"
        )
        if len(catalog) == 0:
            print("  no MaNGA targets in cone — nothing to write", file=sys.stderr)
            return 1

    if args.max_files is not None:
        catalog = catalog[:args.max_files]
        print(f"  capped to {len(catalog)} plate-ifus via --max-files")

    records: list[dict] = []
    for i, row in enumerate(catalog, 1):
        plateifu = _b2s(row["plateifu"])
        cube_file = cube_path(args.raw_root, plateifu)
        map_file = maps_path(args.raw_root, plateifu)
        try:
            rec = process_cube(plateifu, cube_file, map_file)
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            print(f"  [{i}/{len(catalog)}] {plateifu}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        if rec is None:
            print(f"  [{i}/{len(catalog)}] {plateifu}: missing cube or map file",
                  file=sys.stderr)
            continue
        records.append(rec)
        n_maps = len(rec["maps"])
        print(f"  [{i}/{len(catalog)}] {plateifu}: {n_maps} maps")

    if not records:
        print("ERROR: no plate-ifus successfully processed", file=sys.stderr)
        return 1

    print(f"\nBuilding HATS table with {len(records)} records")
    table = build_table(records, catalog)
    print(f"Writing HATS catalog: {table.num_rows} rows")
    catalog_dir = write_hats(
        [table],
        output_path=args.output_root,
        catalog_name=CATALOG_NAME,
        pixel_threshold=args.pixel_threshold,
    )
    print(f"Done: {catalog_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
