"""Verification utilities for HATS catalogs built from MMU v1 HDF5.

Usage:
    from mmu.verify import verify_catalog_against_hdf5
    report = verify_catalog_against_hdf5("path/to/catalog", "path/to/source.hdf5")
    assert report.ok, report.summary()

Or as a CLI:
    python -m mmu.verify --catalog path/to/catalog --source-hdf5 path/to/source.hdf5
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq

from mmu.hats_import import RA_ALIASES, DEC_ALIASES, OBJECT_ID_ALIASES, _resolve_alias


@dataclasses.dataclass
class VerificationReport:
    catalog: str
    source: str
    checks: list[tuple[str, bool, str]] = dataclasses.field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = ""):
        self.checks.append((name, ok, detail))

    @property
    def ok(self) -> bool:
        return all(c[1] for c in self.checks)

    def summary(self) -> str:
        lines = [f"Verification of {self.catalog}", f"  vs source: {self.source}", ""]
        for name, ok, detail in self.checks:
            tag = "OK  " if ok else "FAIL"
            lines.append(f"  [{tag}] {name}{(' — ' + detail) if detail else ''}")
        passed = sum(1 for c in self.checks if c[1])
        lines.append("")
        lines.append(f"Result: {passed}/{len(self.checks)} checks passed")
        return "\n".join(lines)


def _find_hats_dataset_dir(catalog_dir: str) -> str:
    """Locate the dataset/ subdirectory inside a HATS catalog."""
    candidates = [
        os.path.join(catalog_dir, "dataset"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"No dataset/ dir under {catalog_dir}")


def _read_all_parquet(catalog_dir: str, columns: list[str] | None = None):
    import pyarrow.parquet as pq
    dataset_dir = _find_hats_dataset_dir(catalog_dir)
    files = sorted(p for p in _walk(dataset_dir) if p.endswith(".parquet")
                   and "_metadata" not in p and "_common_metadata" not in p)
    tables = [pq.read_table(p, columns=columns) for p in files]
    if not tables:
        raise RuntimeError(f"No parquet files found in {dataset_dir}")
    import pyarrow as pa
    return pa.concat_tables(tables, promote_options="default")


def _walk(d):
    for root, _, files in os.walk(d):
        for f in files:
            yield os.path.join(root, f)


def verify_catalog_against_hdf5(
    catalog_dir: str,
    source_hdf5: str,
    n_rows_used: int | None = None,
) -> VerificationReport:
    """Run a battery of integrity checks on a HATS catalog vs its source HDF5.

    Args:
        catalog_dir: Path to a HATS catalog directory (the inner one with dataset/).
        source_hdf5: Path to the source HDF5 file used to build the catalog.
        n_rows_used: If the catalog was built from only the first N rows of HDF5,
            pass N here. If None, the catalog row count is used as ``n_rows_used``
            (so per-value comparisons stay aligned even when the catalog was a slice).
    """
    report = VerificationReport(catalog=catalog_dir, source=source_hdf5)

    if not os.path.exists(catalog_dir):
        report.add("catalog exists", False, f"missing {catalog_dir}")
        return report
    report.add("catalog exists", True)

    if not os.path.exists(source_hdf5):
        report.add("source exists", False, f"missing {source_hdf5}")
        return report
    report.add("source exists", True)

    hats_table = _read_all_parquet(catalog_dir, columns=None)
    report.add("read parquet", True, f"{hats_table.num_rows} rows, {hats_table.num_columns} cols")

    with h5py.File(source_hdf5, "r") as f:
        keys = list(f.keys())
        ra_key = _resolve_alias(keys, RA_ALIASES)
        dec_key = _resolve_alias(keys, DEC_ALIASES)
        obj_key = _resolve_alias(keys, OBJECT_ID_ALIASES)

        # Row count: if n_rows_used not given, infer from catalog (for slices)
        n_src_total = f[ra_key].shape[0]
        if n_rows_used is None and hats_table.num_rows < n_src_total:
            # Treat the catalog row count as authoritative for the slice
            n_rows_used = hats_table.num_rows
        n_src = min(n_src_total, n_rows_used) if n_rows_used else n_src_total
        report.add(
            "row count matches",
            hats_table.num_rows == n_src,
            f"hats={hats_table.num_rows}, hdf5={n_src}",
        )

        # Required columns present
        has_ra = "ra" in hats_table.schema.names
        has_dec = "dec" in hats_table.schema.names
        has_obj = "object_id" in hats_table.schema.names
        report.add("ra column present", has_ra)
        report.add("dec column present", has_dec)
        if obj_key:
            report.add("object_id column present", has_obj)

        # RA/Dec value match (sorted, since HATS may reorder)
        if has_ra and has_dec:
            src_ra = np.sort(f[ra_key][:n_src])
            src_dec = np.sort(f[dec_key][:n_src])
            hats_ra = np.sort(hats_table.column("ra").to_numpy())
            hats_dec = np.sort(hats_table.column("dec").to_numpy())
            ra_match = np.allclose(src_ra, hats_ra, equal_nan=True)
            dec_match = np.allclose(src_dec, hats_dec, equal_nan=True)
            report.add("ra values match (sorted)", ra_match)
            report.add("dec values match (sorted)", dec_match)

        # RA/Dec ranges sane
        if has_ra:
            ra_arr = hats_table.column("ra").to_numpy()
            ra_ok = (ra_arr >= 0).all() and (ra_arr <= 360).all()
            report.add("ra in [0, 360]", ra_ok,
                       f"min={ra_arr.min():.2f}, max={ra_arr.max():.2f}")
        if has_dec:
            dec_arr = hats_table.column("dec").to_numpy()
            dec_ok = (dec_arr >= -90).all() and (dec_arr <= 90).all()
            report.add("dec in [-90, 90]", dec_ok,
                       f"min={dec_arr.min():.2f}, max={dec_arr.max():.2f}")

        # Spectrum struct, if HDF5 has spectrum_*
        # 3D spectrum_* columns (e.g. DESI spectrum_lsf) are intentionally skipped
        # by the auto-converter, so they shouldn't be required in the struct.
        spec_keys = []
        for k in keys:
            if not k.startswith("spectrum_"):
                continue
            if f[k].ndim >= 3:
                continue  # 3D columns are intentionally not in the spectrum struct
            spec_keys.append(k)
        if spec_keys:
            has_spec = "spectrum" in hats_table.schema.names
            report.add("spectrum struct present", has_spec)
            if has_spec:
                spec_type = hats_table.schema.field("spectrum").type
                fields = [spec_type.field(i).name for i in range(spec_type.num_fields)]
                expected = {k.replace("spectrum_", "") for k in spec_keys}
                missing = expected - set(fields)
                report.add(
                    "spectrum struct has all 1D/2D fields",
                    not missing,
                    f"missing: {missing}" if missing else "",
                )

        # image_array → image_array + image_array_shape
        if "image_array" in keys:
            has_img = "image_array" in hats_table.schema.names
            has_shape = "image_array_shape" in hats_table.schema.names
            report.add("image_array column present", has_img)
            report.add("image_array_shape column present", has_shape)
            if has_img and has_shape:
                # Reconstruct one image, check shape matches HDF5
                flat = np.array(hats_table.column("image_array")[0].as_py())
                shape = list(hats_table.column("image_array_shape")[0].as_py())
                expected_shape = list(f["image_array"].shape[1:])
                report.add(
                    "image shape matches",
                    shape == expected_shape and flat.size == int(np.prod(shape)),
                    f"hats={shape}, hdf5={expected_shape}",
                )

        # Lazy column read smoke test
        try:
            partial = _read_all_parquet(catalog_dir, columns=["ra", "dec"])
            report.add("lazy column read works", partial.num_columns == 2)
        except Exception as e:
            report.add("lazy column read works", False, str(e))

    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Verify a HATS catalog against its source HDF5")
    parser.add_argument("--catalog", required=True, help="Path to HATS catalog directory")
    parser.add_argument("--source-hdf5", required=True, help="Path to source HDF5 file")
    parser.add_argument("--n-rows", type=int, default=None,
                        help="Limit used during conversion (if any)")
    args = parser.parse_args(argv)

    report = verify_catalog_against_hdf5(args.catalog, args.source_hdf5, args.n_rows)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
