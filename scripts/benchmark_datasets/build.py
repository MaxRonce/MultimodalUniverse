"""Build small HATS collections of public evaluation labels."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from mmu.hats_import import write_hats
from mmu.safety import validate_hats_root

HUB_SOURCES = {
    "gz10": (
        "hf://datasets/UniverseTBD/mmu_gz10@e99918709f5c6bccf496bebbc2790cb69c51851c",
        ["gz10_label"],
    ),
    "desi_provabgs": (
        "hf://datasets/UniverseTBD/mmu_desi_provabgs@a50ea5c2baacfea8d88c48baf78bd9c507c2f525",
        ["Z_HP", "LOG_MSTAR", "ZERR", "TSNR2_BGS", "MAG_R"],
    ),
}
DOWNLOADS = {
    "euclid_q1_morphology": (
        "https://zenodo.org/records/15106473/files/morphology_catalogue.parquet",
        "b325012a112df9d20d153934a8b6ceec085768261d5b334b6a9b61879b8b19c6",
    ),
    "galaxy_cruise": (
        "https://galaxycruise.mtk.nao.ac.jp/data/galaxy_cruise_season1.cat.gz",
        "e2eac568d979c2c74f450e70499421734c81c49d799f56c2547bb674b1f1452a",
    ),
    "galaxy_zoo_decals": (
        "https://zenodo.org/records/4573248/files/gz_decals_volunteers_5.parquet",
        "f8959f933d0ad88a8dbb5d1db2d003e131c2fb098778a3d45ad0416995355153",
    ),
}
GZD_QUESTIONS = {
    "smooth-or-featured",
    "disk-edge-on",
    "has-spiral-arms",
    "bar",
    "bulge-size",
    "how-rounded",
    "edge-on-bulge",
    "spiral-winding",
    "spiral-arm-count",
    "merging",
}
EUCLID_COLUMNS = {
    "right_ascension": "ra",
    "declination": "dec",
    "object_id": "object_id",
    "tile_index": "tile_index",
    "warning_galaxy_fails_training_cuts": "warning_galaxy_fails_training_cuts",
    "smooth-or-featured_smooth_fraction": "p_smooth",
    "smooth-or-featured_featured-or-disk_fraction": "p_featured_or_disk",
    "smooth-or-featured_problem_fraction": "p_problem",
    "disk-edge-on_yes_fraction": "p_edge_on_given_disk",
    "has-spiral-arms_yes_fraction": "p_spiral_given_disk_and_not_edge_on",
}
CRUISE_COLUMNS = {
    0: "object_id",
    1: "ra",
    2: "dec",
    3: "specz",
    4: "zcmodel_mag",
    7: "layer",
    8: "num_class",
    9: "p_spiral",
    10: "p_interaction",
    15: "galaxy_type_valid_fraction",
    16: "interaction_valid_fraction",
}


def read_labels(name: str, scratch: Path) -> tuple[pa.Table, str]:
    if name in HUB_SOURCES:
        import lsdb

        uri, labels = HUB_SOURCES[name]
        catalog = lsdb.open_catalog(uri, columns=["ra", "dec", "object_id", *labels])
        return pa.Table.from_pandas(catalog.compute(), preserve_index=False), uri

    uri, sha256 = DOWNLOADS[name]
    source = scratch / uri.rsplit("/", 1)[-1]
    with requests.get(uri, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with source.open("wb") as download:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                download.write(chunk)
    with source.open("rb") as downloaded:
        digest = hashlib.file_digest(downloaded, "sha256").hexdigest()
    if digest != sha256:
        raise ValueError(f"{name}: {uri} does not match the pinned SHA-256")
    if name == "euclid_q1_morphology":
        table = pq.read_table(source, columns=list(EUCLID_COLUMNS))
        return table.rename_columns(EUCLID_COLUMNS), uri
    if name == "galaxy_zoo_decals":
        parquet = pq.ParquetFile(source)
        votes = [col for col in parquet.schema_arrow.names if col.split("_", 1)[0] in GZD_QUESTIONS]
        table = parquet.read(columns=["ra", "dec", "iauname", *votes])
        return table.rename_columns({"iauname": "object_id"}), uri

    frame = pd.read_csv(
        source, sep=r"\s+", header=None, usecols=list(CRUISE_COLUMNS), dtype={0: "string"}
    )
    return pa.Table.from_pandas(frame.rename(columns=CRUISE_COLUMNS), preserve_index=False), uri


def build(name: str, root: Path, workers: int) -> None:
    validate_hats_root(str(root), benchmark=True)
    root.mkdir(parents=True, exist_ok=True)
    output = root / name
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(prefix=f".{name}-", dir=root) as staging:
        staging = Path(staging)
        table, source = read_labels(name, staging)
        for field, dtype in (("ra", pa.float64()), ("dec", pa.float64()), ("object_id", pa.string())):
            table = table.set_column(
                table.schema.get_field_index(field), field, pc.cast(table[field], dtype)
            )
        ra, dec = (table[field].to_numpy() for field in ("ra", "dec"))
        if not np.all((0 <= ra) & (ra < 360) & (-90 <= dec) & (dec <= 90)):
            raise ValueError(f"{name}: invalid sky coordinates")
        ids = table["object_id"]
        if not len(table) or ids.null_count or pc.count_distinct(ids).as_py() != len(table):
            raise ValueError(f"{name}: expected nonempty, unique object IDs")

        inner = Path(write_hats([table], str(staging), name, n_workers=workers, debug=workers == 1))
        collection = inner.parent
        metadata = {"source": source, "rows": len(table), "columns": table.column_names}
        if name in DOWNLOADS:
            metadata["sha256"] = DOWNLOADS[name][1]
        (collection / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        collection.rename(output)
    print(f"{name}: {len(table):,} labels -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=[*HUB_SOURCES, *DOWNLOADS])
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build(args.name, args.output_root, args.workers)
