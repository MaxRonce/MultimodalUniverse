"""Write ECLIPSE matched-system counts as CSV, including empty survey/split cells."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def report(path: Path) -> None:
    grades = ("A", "B", "C", "X")
    splits = ("train", "validation", "test")
    parquet = pq.ParquetFile(path)
    metadata = json.loads(parquet.schema_arrow.metadata[b"benchmark"])
    if metadata["catalog"] != "euclid_dr1_eclipse":
        raise ValueError("Expected the ECLIPSE match parquet")
    surveys = metadata["surveys"]
    systems = defaultdict(set)
    for row in parquet.read(columns=["object_id", "expert_grade", "survey", "split"]).to_pylist():
        survey, split, grade = row["survey"], row["split"], row["expert_grade"]
        if survey not in surveys or split not in splits or grade not in grades:
            raise ValueError(f"Unexpected match row: {row}")
        systems[survey, split, grade].add(row["object_id"])

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("survey", "split", *grades, "total"))
    for survey in surveys:
        for split in splits:
            counts = [len(systems[survey, split, grade]) for grade in grades]
            writer.writerow((survey, split, *counts, sum(counts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matches", type=Path)
    report(parser.parse_args().matches)
