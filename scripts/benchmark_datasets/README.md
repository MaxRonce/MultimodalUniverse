# Benchmark datasets

The `benchmark_datasets` target builds four small HATS collections with sky
coordinates, object IDs, and labels. Each collection includes a 10″ margin
and a `source.json` with the upstream revision or checksum. The target is
separate from the regular MMU builds.

| Catalog | Labels | First AION tables to check |
| --- | --- | --- |
| [Galaxy10 DECaLS](https://huggingface.co/datasets/UniverseTBD/mmu_gz10) | 10 morphology classes | Legacy South and HSC Wide |
| [DESI PROVABGS](https://huggingface.co/datasets/UniverseTBD/mmu_desi_provabgs) | Spectroscopic redshift and fitted stellar mass | Legacy South and DESI DR1 |
| [Galaxy Cruise S1](https://galaxycruise.mtk.nao.ac.jp/en/dr_season1.html) | Volunteer spiral and interaction probabilities, with vote counts | HSC Wide and HSC Deep |
| [Euclid Q1 morphology](https://zenodo.org/records/15106473) | Zoobot morphology probabilities and MER object ID | Euclid Q1 |

Galaxy10 and PROVABGS are read from public MMU HATS catalogs with only the
benchmark columns selected. Galaxy Cruise (2.3 MB) and Euclid Q1 (97 MB) are
downloaded from their release sites. These are derived label catalogs built
with MMU's `write_hats` helper. The build retains source values and nulls,
checks coordinates and release-local IDs, and writes the finished collection
from a staging directory.

## Sources and label conventions

| Catalog | Release and citation | Fields used in the benchmark |
| --- | --- | --- |
| Galaxy10 DECaLS | MMU HATS at the revision in `source.json`. Cite the [Galaxy10 release](https://doi.org/10.5281/zenodo.10845026). | `gz10_label` is the published integer class, 0–9. `object_id` is local to the Galaxy10 catalog. [Class definitions](https://astronn.readthedocs.io/en/latest/galaxy10.html) |
| DESI PROVABGS | MMU HATS of the [DESI EDR/SV3 value-added catalog](https://data.desi.lbl.gov/doc/releases/edr/vac/provabgs/). [Hahn et al.](https://doi.org/10.3847/1538-4357/ac8983) | `Z_HP` and `ZERR` come from Redrock. `LOG_MSTAR` is the fitted log stellar mass from `PROVABGS_LOGMSTAR_BF`. `TSNR2_BGS` and `MAG_R` retain the source quality and magnitude information. |
| Galaxy Cruise | [Season 1, HSC PDR2](https://galaxycruise.mtk.nao.ac.jp/en/dr_season1.html). [Tanaka et al. 2023](https://doi.org/10.1093/pasj/psad055) | `p_spiral` and `p_interaction` are the released corrected vote probabilities. `num_class` and the two valid-vote fractions come from the same release. |
| Euclid Q1 | [Q1_R1 morphology, Zenodo v4](https://zenodo.org/records/15106473). [Walmsley et al.](https://arxiv.org/abs/2503.15310) | The `p_*` columns contain the published Zoobot vote fractions. The disk and spiral columns are conditional and retain the source nulls. `object_id` is the MER identifier. |

The column mappings are in [`build.py`](build.py). Use finite labels for each
Euclid task and report that task's sample size. The source flag
`warning_galaxy_fails_training_cuts` marks galaxies in a range where the
release reports potentially lower model reliability. It is carried through
to let the evaluation choose and report its selection.

## Build the catalogs

From an installed MMU checkout on Rusty:

```bash
ROOT="$HOME/ceph/aion2/benchmark-datasets"
srun -p genx -n 1 -c 4 --mem=16G --time=02:00:00 \
    .venv/bin/snakemake --cores 4 benchmark_datasets \
    --config "benchmark_root=$ROOT"
```

Replace `benchmark_datasets` with `"$ROOT/gz10"` to build one collection.
Without `benchmark_root`, the target writes under `<hats_root>/benchmarks`.
On shared Ceph, this fork also accepts
`/mnt/ceph/users/polymathic/mronceray/benchmark-datasets` for benchmark builds.

## Match raw survey catalogs

Use the raw HATS collection for the relevant release. The right-hand margin
lets LSDB find counterparts across HATS partition edges.

```python
import lsdb

labels = lsdb.open_catalog("/path/to/benchmark-datasets/gz10")
images = lsdb.open_catalog("/path/to/legacy_south", columns=["object_id", "ra", "dec"])
matches = labels.crossmatch(
    images, radius_arcsec=1.0, n_neighbors=1, require_right_margin=True
).compute()
```

## Match the tables used by an AION checkpoint

Run from the MMU checkout with the AION Python environment. Use the resolved
`config.yaml` saved in the training run directory to reuse that run's exact sky
split. Before training, use the eight-table AION recipe at
`configs/tutorials/training/data.yaml` and pass the directory containing the
`.lance` tables as `--tokens-dir`. The tables need the spatial index used by
the AION checkout. The command uses AION's nearest-neighbor crossmatch at 1″.

```bash
AION="$HOME/dev/AION-2"
AION_CONFIG="/path/to/aion/run/config.yaml"
OUT="$ROOT/matches-20261118"

for catalog in gz10 desi_provabgs galaxy_cruise euclid_q1_morphology; do
    srun -p genx -n 1 -c 4 --mem=16G --time=04:00:00 \
        "$AION/.venv/bin/python" scripts/benchmark_datasets/match_aion.py \
        "$ROOT/$catalog" "$AION_CONFIG" "$OUT/$catalog.parquet" \
        --expected-tables 8 || break
done
```

The script checks every table in the configuration by default. Each Parquet
row contains the label ID, survey row number, coordinates, separation and
survey object ID when the table supplies it. Its `benchmark` metadata records
the Lance paths and versions read, source release and split counts, including
zeros. Guard rows are counted and left out of the Parquet rows. The counts
describe geometric matches by sky split. Token availability for a particular
evaluation task can be checked when reading the matched rows.

Print the train, validation and test counts from the four reports:

```bash
"$AION/.venv/bin/python" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

from pyarrow.parquet import read_schema

root = Path(sys.argv[1])
print(f"{'Benchmark':24} {'Survey':22} {'Train':>10} {'Val':>10} {'Test':>10}")
for catalog in ("gz10", "desi_provabgs", "galaxy_cruise", "euclid_q1_morphology"):
    report = json.loads(read_schema(root / f"{catalog}.parquet").metadata[b"benchmark"])
    for survey, info in report["surveys"].items():
        c = info["counts"]
        print(f"{catalog:24} {survey:22} {c['train']:10,} {c['validation']:10,} {c['test']:10,}")
PY
```

Fit a probe on observed tokens from the train region and choose settings on
validation. Compare observed and generated target tokens on the same labeled
test objects with the probe frozen. For a translation pair, join the matches
by benchmark `object_id` and `split`, then require both survey detections to
be within 1″ of each other, too.
