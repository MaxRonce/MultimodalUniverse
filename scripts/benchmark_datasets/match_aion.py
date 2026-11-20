"""Attach public benchmark labels to the token tables in a saved AION run."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from aion2.data.loader.config import LiveCrossmatchConfig, SplitConfig, TokenTableConfig
from aion2.data.loader.live_crossmatch import LiveCrossmatch
from aion2.data.loader.splits import RowRuns, in_split
from aion2.data.loader.token_table import TokenTable
from aion2.token_tables.format import CATALOG_NAMESPACE_KEY, attach_namespace


RADIUS_ARCSEC = 1.0
COORDINATES = ("ra", "dec", "healpix_29")


def init_args(config: dict) -> dict:
    return config.get("init_args", config)


def load_mixture(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = init_args(config.get("trainer", config))
    for section in ("data_config", "mixture_dataset_config"):
        config = init_args(config[section])
    return config


def resolve_table_paths(
    mixture: dict, surveys: list[str] | None, tokens_dir: Path | None
) -> dict[str, str]:
    tables = mixture["token_table_configs"]
    paths = {}
    for survey in surveys or tables:
        path = init_args(tables[survey])["table_path"]
        if "{TOKENS_DIR}" in path:
            if tokens_dir is None:
                raise ValueError("Pass --tokens-dir for a training data.yaml.")
            path = path.replace("{TOKENS_DIR}", str(tokens_dir.resolve()))
        paths[survey] = path
    return paths


def split_names(healpix: np.ndarray, config: SplitConfig) -> np.ndarray:
    names = np.empty(len(healpix), dtype=object)
    for name in ("train", *config.holdout_healpix):
        names[in_split(healpix, replace(config, split=name))] = name
    return names


def separation_arcsec(left: pa.Table, right: pa.Table) -> np.ndarray:
    ra_left, dec_left = (np.deg2rad(left[c].to_numpy()) for c in ("ra", "dec"))
    ra_right, dec_right = (np.deg2rad(right[c].to_numpy()) for c in ("ra", "dec"))
    haversine = (
        np.sin((dec_right - dec_left) / 2) ** 2
        + np.cos(dec_left) * np.cos(dec_right) * np.sin((ra_right - ra_left) / 2) ** 2
    )
    return 7200 * np.rad2deg(np.arcsin(np.sqrt(np.clip(haversine, 0, 1))))


def match_survey(
    labels: pa.Table, anchor: TokenTable, survey: str, target: TokenTable, split: SplitConfig
) -> tuple[pa.Table, dict[str, int]]:
    matcher = LiveCrossmatch(
        survey,
        LiveCrossmatchConfig("labels", survey, RADIUS_ARCSEC, anchors_per_query=128),
        anchor=anchor,
        target=target,
        anchor_rows=RowRuns.every_row(len(labels)),
        split_config=None,
        seed=0,
    )
    rows = matcher.match(np.arange(len(labels), dtype=np.int64))
    found = np.flatnonzero(rows >= 0)
    columns = list(COORDINATES)
    if attach_namespace(target.catalog, "object_id") in target.dataset.schema.names:
        columns.append("object_id")
    targets = target.dataset.take(
        rows[found].tolist(), columns=[attach_namespace(target.catalog, c) for c in columns]
    ).rename_columns(columns)
    selected = labels.take(found)
    label_splits = split_names(selected["_healpix_29"].to_numpy(), split)
    target_splits = split_names(targets["healpix_29"].to_numpy(), split)

    distance = separation_arcsec(selected, targets)
    accepted = np.flatnonzero(label_splits == target_splits)
    # Several labels can pick one survey row. Keep the closest, breaking ties by label ID.
    order = np.lexsort((
        selected["object_id"].to_numpy()[accepted], distance[accepted], rows[found][accepted]
    ))
    best = accepted[order]
    _, first = np.unique(rows[found][best], return_index=True)
    kept = best[first]
    counts = {
        name: int((label_splits[kept] == name).sum())
        for name in ("train", *split.holdout_healpix)
    }
    counts["cross_split"] = len(found) - len(accepted)
    counts["duplicate_target"] = len(accepted) - len(kept)
    kept = kept[label_splits[kept] != "guard"]

    result = selected.take(kept)
    survey_ids = pa.nulls(len(kept), type=pa.string())
    if "object_id" in columns:
        survey_ids = pc.cast(targets["object_id"].take(kept), pa.string())
    fields = {
        "survey": pa.array([survey] * len(kept), type=pa.string()),
        "split": pa.array(label_splits[kept], type=pa.string()),
        "survey_row": pa.array(rows[found][kept]),
        "survey_object_id": survey_ids,
        "survey_ra": targets["ra"].take(kept),
        "survey_dec": targets["dec"].take(kept),
        "separation_arcsec": pa.array(distance[kept]),
    }
    for name, values in fields.items():
        result = result.append_column(name, values)
    return result, counts


def match(
    catalog: Path,
    config_path: Path,
    output: Path,
    surveys: list[str] | None,
    *,
    tokens_dir: Path | None = None,
    expected_tables: int | None = None,
) -> None:
    mixture = load_mixture(config_path)
    if not mixture.get("split_config"):
        raise ValueError("Use an AION configuration with a spatial split.")
    split = SplitConfig(**init_args(mixture["split_config"]))
    table_paths = resolve_table_paths(mixture, surveys, tokens_dir)
    num_tables = len(set(table_paths.values()))
    if expected_tables is not None and num_tables != expected_tables:
        raise ValueError(f"Expected {expected_tables} distinct Lance tables, found {num_tables}.")
    labels = pq.read_table(catalog / catalog.name / "dataset", partitioning=None)
    labels = labels.sort_by("_healpix_29")
    source = json.loads((catalog / "source.json").read_text(encoding="utf-8"))
    metadata = {"catalog": catalog.name, "source": source, "split": vars(split), "surveys": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="aion-benchmark-") as scratch:
        anchor_path = str(Path(scratch) / "labels.lance")
        coords = labels.select(["ra", "dec", "_healpix_29"]).rename_columns(
            [attach_namespace("labels", name) for name in COORDINATES]
        )
        lance.write_dataset(
            coords.replace_schema_metadata({CATALOG_NAMESPACE_KEY: "labels"}), anchor_path
        )
        anchor = TokenTable(TokenTableConfig(anchor_path, []), token_specs={})
        results = []
        for survey, table_path in table_paths.items():
            target = TokenTable(TokenTableConfig(table_path, []), token_specs={})
            result, counts = match_survey(labels, anchor, survey, target, split)
            results.append(result)
            metadata["surveys"][survey] = {
                "table_path": table_path, "version": target.dataset.version, "counts": counts
            }
            print(f"{survey}  " + "  ".join(f"{k}={v:,}" for k, v in counts.items()), flush=True)

    metadata["radius_arcsec"] = RADIUS_ARCSEC
    result = pa.concat_tables(results).replace_schema_metadata({"benchmark": json.dumps(metadata)})
    with TemporaryDirectory(dir=output.parent) as staging:
        parquet = Path(staging) / output.name
        pq.write_table(result, parquet)
        parquet.replace(output)
    print(f"{len(result):,} matches -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path, help="MMU benchmark HATS collection")
    parser.add_argument("config", type=Path, help="saved run config.yaml or training data.yaml")
    parser.add_argument("output", type=Path, help="output Parquet with one row per label/survey")
    parser.add_argument("--surveys", nargs="+", help="default: every token table in the run")
    parser.add_argument("--tokens-dir", type=Path, help="resolve {TOKENS_DIR} in training data.yaml")
    parser.add_argument("--expected-tables", type=int, help="require this many distinct Lance paths")
    args = parser.parse_args()
    match(
        args.catalog,
        args.config,
        args.output,
        args.surveys,
        tokens_dir=args.tokens_dir,
        expected_tables=args.expected_tables,
    )
