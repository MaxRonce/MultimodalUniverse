"""Count labeled galaxies observed by two AION surveys within 1 arcsecond."""

import argparse
import csv
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from match_aion import RADIUS_ARCSEC, separation_arcsec

SPLITS = ("train", "validation", "test")
KEYS = ["object_id", "split"]
COORDS = ["survey_ra", "survey_dec"]


def count_pairs(paths: list[Path], *, expected_surveys: int) -> list[tuple]:
    by_survey = {}
    signature = None
    for path in paths:
        table = pq.read_table(path, columns=["survey", *KEYS, *COORDS])
        report = json.loads(table.schema.metadata[b"benchmark"])
        current = tuple(
            report[key] for key in ("catalog", "source", "split", "radius_arcsec")
        )
        if signature is not None and signature != current:
            raise ValueError(
                f"{path}: the input files need the same catalog, source and split"
            )
        if report["radius_arcsec"] != RADIUS_ARCSEC:
            raise ValueError(
                f"{path}: expected crossmatches at {RADIUS_ARCSEC:g} arcsec"
            )
        signature = current
        # Use the metadata to include surveys whose crossmatch returned no rows.
        for survey in report["surveys"]:
            if survey in by_survey:
                raise ValueError(f"{path}: repeated survey {survey!r}")
            by_survey[survey] = table.filter(pc.equal(table["survey"], survey)).drop(
                ["survey"]
            )

    if len(by_survey) != expected_surveys:
        raise ValueError(
            f"Expected {expected_surveys} surveys, found {len(by_survey)}."
        )

    results = []
    for survey_a, survey_b in combinations(sorted(by_survey), 2):
        pairs = by_survey[survey_a].join(
            by_survey[survey_b],
            keys=KEYS,
            join_type="inner",
            left_suffix="_a",
            right_suffix="_b",
        )
        left, right = (
            pairs.select([f"{coord}_{side}" for coord in COORDS]).rename_columns(
                ["ra", "dec"]
            )
            for side in ("a", "b")
        )
        close = separation_arcsec(left, right) <= RADIUS_ARCSEC
        counts = Counter(pairs["split"].to_numpy()[close])
        results.append(
            (signature[0], survey_a, survey_b, *(counts[name] for name in SPLITS))
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "matches", type=Path, nargs="+", help="one catalog's match_aion Parquet files"
    )
    parser.add_argument("--expected-surveys", type=int, default=8)
    args = parser.parse_args()
    rows = count_pairs(args.matches, expected_surveys=args.expected_surveys)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("benchmark", "survey_a", "survey_b", *SPLITS))
    writer.writerows(rows)
