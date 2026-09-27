"""Publish review PNGs to W&B without loading MMU or modifying its environment.

Requires wandb, numpy and Pillow, but no FITS/HATS/PyTorch dependencies. Only
the review panels and summary metrics are uploaded, not the raw NPZ cutouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def publish_review(directory: Path, project: str, entity: str, mode: str) -> None:
    import wandb

    directory = directory.resolve()
    review = json.loads((directory / "review.json").read_text())
    if review["status"] != "PASS":
        raise ValueError("review did not pass")
    panels = []
    for item in review["items"]:
        path = (directory / item["png"]).resolve()
        if path.parent != directory or path.suffix != ".png" or not path.is_file():
            raise ValueError(f"invalid review panel: {path}")
        panels.append(path)
    with wandb.init(
        project=project,
        entity=entity,
        job_type="dataset-review",
        dir=str(directory),
        mode=mode,
        config={key: review[key] for key in ("seed", "sampling", "rendering", "scope")},
    ) as run:
        table = wandb.Table(
            columns=[
                "object_id",
                "i_mag",
                "Re_arcsec",
                "clean_fraction",
                "bands_present",
                "panels",
            ]
        )
        for item, panel in zip(review["items"], panels):
            table.add_data(
                item["object_id"],
                item["i_cModelMag"],
                item["sersic_reff_major"],
                item["clean_fraction"],
                "".join(
                    band
                    for band, present in zip("ugrizy", item["band_present"])
                    if present
                ),
                wandb.Image(str(panel)),
            )
        run.log({"cutouts": table})
        run.summary.update(
            {
                "dataset_rows": review["dataset_rows"],
                "sample_rows": review["sample_rows"],
                "review_status": "PASS",
                "validation_scope": review["scope"],
            }
        )
        print(f"W&B: {run.url if mode == 'online' else run.dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", required=True, type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    publish_review(args.review_dir, args.project, args.entity, args.mode)


if __name__ == "__main__":
    main()
