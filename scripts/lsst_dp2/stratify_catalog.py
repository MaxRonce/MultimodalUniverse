"""Create a deterministic DP2 galaxy sample stratified by magnitude and size."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from astropy.table import Column, Table

REQUIRED_COLUMNS = (
    "objectId",
    "tract",
    "patch",
    "i_cModelMag",
    "i_extendedness",
    "griz_model_extendedness",
    "sersic_reff_major",
    "sersic_no_data_flag",
    "sersic_unknown_flag",
)


def parse_edges(value: str, *, allow_infinite_last: bool = False) -> np.ndarray:
    """Parse strictly increasing comma-separated bin edges."""
    edges = np.asarray([float(item.strip()) for item in value.split(",")], dtype=float)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin edges must contain at least two increasing values")
    if not np.all(np.isfinite(edges[:-1])):
        raise ValueError("only the final bin edge may be infinite")
    if not np.isfinite(edges[-1]) and (
        not allow_infinite_last or not np.isposinf(edges[-1])
    ):
        raise ValueError("only a positive infinite final edge is allowed")
    return edges


def stable_score(object_id: object, seed: int) -> int:
    """Return a platform-independent pseudo-random score for an object ID."""
    payload = f"{seed}:{object_id}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _filled(table: Table, name: str, fill, dtype) -> np.ndarray:
    return np.asarray(np.ma.filled(table[name], fill), dtype=dtype)


def select_stratified(
    table: Table,
    mag_edges: np.ndarray,
    size_edges: np.ndarray,
    per_cell: int,
    seed: int,
    min_model_extendedness: float,
) -> tuple[Table, dict]:
    """Select up to ``per_cell`` valid galaxies from every 2D bin."""
    missing = [name for name in REQUIRED_COLUMNS if name not in table.colnames]
    if missing:
        raise ValueError(f"catalog is missing required columns: {missing}")
    if per_cell <= 0:
        raise ValueError("per_cell must be positive")

    mag = _filled(table, "i_cModelMag", np.nan, float)
    size = _filled(table, "sersic_reff_major", np.nan, float)
    extended = _filled(table, "i_extendedness", np.nan, float)
    model_extended = _filled(table, "griz_model_extendedness", np.nan, float)
    no_data = _filled(table, "sersic_no_data_flag", True, bool)
    unknown = _filled(table, "sersic_unknown_flag", True, bool)
    valid = (
        np.isfinite(mag)
        & np.isfinite(size)
        & np.isfinite(extended)
        & np.isfinite(model_extended)
        & (extended == 1)
        & (model_extended >= min_model_extendedness)
        & ~no_data
        & ~unknown
    )

    mag_bin = np.searchsorted(mag_edges, mag, side="right") - 1
    size_bin = np.searchsorted(size_edges, size, side="right") - 1
    selected_indices: list[int] = []
    selected_labels: dict[int, tuple[str, str]] = {}
    cells = []
    for mi in range(len(mag_edges) - 1):
        for si in range(len(size_edges) - 1):
            candidates = np.flatnonzero(
                valid
                & (mag_bin == mi)
                & (size_bin == si)
            )
            ranked = sorted(
                candidates,
                key=lambda index: stable_score(table["objectId"][index], seed),
            )
            chosen = ranked[:per_cell]
            mag_label = f"[{mag_edges[mi]:g},{mag_edges[mi + 1]:g})"
            size_label = f"[{size_edges[si]:g},{size_edges[si + 1]:g})"
            for index in chosen:
                selected_labels[index] = (mag_label, size_label)
            selected_indices.extend(chosen)
            cells.append(
                {
                    "mag_bin": mag_label,
                    "size_bin_arcsec": size_label,
                    "available": len(candidates),
                    "selected": len(chosen),
                }
            )

    selected = table[selected_indices]
    selected.add_column(
        Column([selected_labels[index][0] for index in selected_indices]),
        name="sample_mag_bin",
    )
    selected.add_column(
        Column([selected_labels[index][1] for index in selected_indices]),
        name="sample_size_bin_arcsec",
    )
    selected.sort(["tract", "patch", "objectId"])
    report = {
        "input_rows": len(table),
        "eligible_rows": int(valid.sum()),
        "selected_rows": len(selected),
        "per_cell_requested": per_cell,
        "seed": seed,
        "min_griz_model_extendedness": min_model_extendedness,
        "mag_edges": mag_edges.tolist(),
        "size_edges_arcsec": [
            float(value) if np.isfinite(value) else None for value in size_edges
        ],
        "cells": cells,
    }
    return selected, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mag-edges", default="20,21,22,23,24")
    parser.add_argument("--size-edges", default="0.4,0.6,1.0,1.5,inf")
    parser.add_argument("--per-cell", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--min-model-extendedness", type=float, default=0.8)
    parser.add_argument("--require-full-cells", action="store_true")
    args = parser.parse_args(argv)

    try:
        mag_edges = parse_edges(args.mag_edges)
        size_edges = parse_edges(args.size_edges, allow_infinite_last=True)
        table = Table.read(args.catalog, format="parquet")
        selected, report = select_stratified(
            table,
            mag_edges,
            size_edges,
            args.per_cell,
            args.seed,
            args.min_model_extendedness,
        )
        underfilled = [
            cell for cell in report["cells"] if cell["selected"] < args.per_cell
        ]
        if underfilled and args.require_full_cells:
            raise RuntimeError(f"{len(underfilled)} sample cells are underfilled")

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        selected.write(temporary, format="parquet", overwrite=True)
        os.replace(temporary, output)
        report_path = Path(str(output) + ".selection.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        print(f"Wrote {len(selected)} objects to {output}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
