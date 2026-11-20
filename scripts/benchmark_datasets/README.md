# Benchmark datasets

The `benchmark_datasets` target builds five small HATS collections with sky
coordinates, object IDs, and labels. Each collection includes a 10″ margin
and a `source.json` with the upstream revision or checksum. The target is
separate from the regular MMU builds.

| Catalog | Labels | First AION tables to check |
| --- | --- | --- |
| [Galaxy10 DECaLS](https://huggingface.co/datasets/UniverseTBD/mmu_gz10) | 10 morphology classes | Legacy South and HSC Wide |
| [DESI PROVABGS](https://huggingface.co/datasets/UniverseTBD/mmu_desi_provabgs) | Spectroscopic redshift and fitted stellar mass | Legacy South and DESI DR1 |
| [Galaxy Cruise S1](https://galaxycruise.mtk.nao.ac.jp/en/dr_season1.html) | Volunteer spiral and interaction probabilities, with vote counts | HSC Wide and HSC Deep |
| [Euclid Q1 morphology](https://zenodo.org/records/15106473) | Zoobot morphology probabilities and MER object ID | Euclid Q1 |
| [Galaxy Zoo DECaLS GZD-5](https://zenodo.org/records/4573248) | Volunteer vote counts and fractions for 10 morphology questions | Legacy South, then every AION table |

Galaxy10 and PROVABGS are read from public MMU HATS catalogs with only the
benchmark columns selected. Galaxy Cruise (2.3 MB), Euclid Q1 (97 MB) and Galaxy
Zoo DECaLS (40.5 MB) are downloaded from their release sites. MMU's
`write_hats` helper builds the derived catalogs. The build retains source values and nulls,
checks coordinates and release-local IDs, and writes the finished collection
from a staging directory.

## Sources and label conventions

| Catalog | Release and citation | Fields used in the benchmark |
| --- | --- | --- |
| Galaxy10 DECaLS | MMU HATS at the revision in `source.json`. Cite the [Galaxy10 release](https://doi.org/10.5281/zenodo.10845026). | `gz10_label` is the published integer class, 0–9. `object_id` is local to the Galaxy10 catalog. [Class definitions](https://astronn.readthedocs.io/en/latest/galaxy10.html) |
| DESI PROVABGS | MMU HATS of the [DESI EDR/SV3 value-added catalog](https://data.desi.lbl.gov/doc/releases/edr/vac/provabgs/). [Hahn et al.](https://doi.org/10.3847/1538-4357/ac8983) | `Z_HP` and `ZERR` come from Redrock. `LOG_MSTAR` is the fitted log stellar mass from `PROVABGS_LOGMSTAR_BF`. `TSNR2_BGS` and `MAG_R` retain the source quality and magnitude information. |
| Galaxy Cruise | [Season 1, HSC PDR2](https://galaxycruise.mtk.nao.ac.jp/en/dr_season1.html). [Tanaka et al. 2023](https://doi.org/10.1093/pasj/psad055) | `p_spiral` and `p_interaction` are the released corrected vote probabilities. `num_class` and the two valid-vote fractions come from the same release. |
| Euclid Q1 | [Q1_R1 morphology, Zenodo v4](https://zenodo.org/records/15106473). [Walmsley et al.](https://arxiv.org/abs/2503.15310) | The `p_*` columns contain the published Zoobot vote fractions. The disk and spiral columns are conditional and retain the source nulls. `object_id` is the MER identifier. |
| Galaxy Zoo DECaLS | [GZD-5 volunteer classifications, Zenodo v0.0.2](https://zenodo.org/records/4573248). [Walmsley et al.](https://arxiv.org/abs/2102.08414) | `object_id` is the published `iauname`. The 10 questions keep their original vote counts, fractions and debiased fields. |

The column mappings are in [`build.py`](build.py). Use finite labels for each
Euclid task and report that task's sample size. The source flag
`warning_galaxy_fails_training_cuts` marks galaxies in a range where the
release reports potentially lower model reliability. It is carried through
to let the evaluation choose and report its selection.

GZD-5 provides votes for [fine-grained morphology](https://github.com/PolymathicAI/astrobench/blob/main/docs/reference/task-catalog.md#gzd-5-fine-grained-morphology),
and the `has-spiral-arms_yes_fraction` and `merging_merger_fraction` labels
for [spiral](https://github.com/PolymathicAI/astrobench/blob/main/docs/reference/task-catalog.md#galaxy-zoo-spirals)
and [merger retrieval](https://github.com/PolymathicAI/astrobench/blob/main/docs/reference/task-catalog.md#galaxy-zoo-mergers).
The fractions follow the survey's decision tree. For morphology, evaluate a
question on galaxies with votes for that question. For retrieval, AstroBench
uses at least three root votes and a fraction of at least 0.9 to choose
queries. This build uses AION's HEALPix splits. AstroBench's published
morphology benchmark uses the GZ Evo cohort and its release split.

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

for catalog in gz10 desi_provabgs galaxy_cruise euclid_q1_morphology galaxy_zoo_decals; do
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

Print the train, validation and test counts from the five reports:

```bash
"$AION/.venv/bin/python" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

from pyarrow.parquet import read_schema

root = Path(sys.argv[1])
print(f"{'Benchmark':24} {'Survey':22} {'Train':>10} {'Val':>10} {'Test':>10}")
for catalog in ("gz10", "desi_provabgs", "galaxy_cruise", "euclid_q1_morphology", "galaxy_zoo_decals"):
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

## Count labeled pairs on Rusty

Each row of the report gives a benchmark catalog, two surveys and three sky
split counts. One galaxy contributes to a pair when both surveys matched its
label in the same split and their detections are within 1″ of each other.
All 28 survey pairs appear, including zeros. The counts use the two recorded
counterparts for each label and measure geometric coverage.

For a catalog already matched to all eight tables, the count needs only its
Parquet file.

```bash
"$AION/.venv/bin/python" scripts/benchmark_datasets/count_pairs.py \
    "$OUT/desi_provabgs.parquet" > "$OUT/desi_provabgs-pairs.csv"
```

Build just GZD-5 if the other catalogs already exist. Reuse the same split
configuration that produced their existing match Parquets.

```bash
MMU="$HOME/MultimodalUniverse"
TOKENS_DIR=/mnt/ceph/users/polymathic/AION-2-tokens/aion2.lancedb
GZD_MATCHES="$ROOT/gzd5-matches-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$GZD_MATCHES"
srun -p genx -n 1 -c 4 --mem=16G --time=02:00:00 --chdir="$MMU" \
    .venv/bin/snakemake --cores 4 "$ROOT/galaxy_zoo_decals" \
    --config "benchmark_root=$ROOT"

export MMU AION AION_CONFIG ROOT TOKENS_DIR GZD_MATCHES
GZD_JOB=$(sbatch --parsable --job-name=gzd5-match -p genx -n 1 -c 4 --mem=16G --time=04:00:00 \
    --array=0-7%4 --output="$GZD_MATCHES/%A_%a.log" --wrap='
    set -eu
    set -- euclid_q1 hsc_deep hsc_wide legacysurvey_south lsst_dp2 jades_jwst desi_dr1 sdss
    shift "$SLURM_ARRAY_TASK_ID"
    survey=$1
    "$AION/.venv/bin/python" "$MMU/scripts/benchmark_datasets/match_aion.py" \
        "$ROOT/galaxy_zoo_decals" "$AION_CONFIG" "$GZD_MATCHES/$survey.parquet" \
        --tokens-dir "$TOKENS_DIR" --surveys "$survey" --expected-tables 1
')
GZD_JOB=${GZD_JOB%%;*}
squeue -j "$GZD_JOB"
```

After the GZD-5 array finishes, `grep -h 'train=' "$GZD_MATCHES"/*.log` shows
the counts for each survey on its own.

Run the five pair counts in parallel, using `OUT` from the four previous
matches. The dependency waits for all eight GZD-5 matches to finish.

```bash
PAIR_OUT="$ROOT/pairs-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$PAIR_OUT"
export OUT PAIR_OUT
PAIR_JOB=$(sbatch --parsable --job-name=benchmark-pairs -p genx -n 1 -c 2 --mem=8G --time=00:30:00 \
    --array=0-4%4 --dependency="afterok:$GZD_JOB" --output="$PAIR_OUT/%A_%a.log" --wrap='
    set -eu
    set -- gz10 desi_provabgs galaxy_cruise euclid_q1_morphology galaxy_zoo_decals
    shift "$SLURM_ARRAY_TASK_ID"
    catalog=$1
    if [ "$catalog" = galaxy_zoo_decals ]; then
        set -- "$GZD_MATCHES"/*.parquet
    else
        set -- "$OUT/$catalog.parquet"
    fi
    "$AION/.venv/bin/python" "$MMU/scripts/benchmark_datasets/count_pairs.py" "$@" \
        > "$PAIR_OUT/$catalog.csv"
')
PAIR_JOB=${PAIR_JOB%%;*}
sacct -j "$GZD_JOB,$PAIR_JOB" -X --format=JobID,State,Elapsed,ExitCode
```

Once both arrays are complete, merge the reports and show pairs with held-out
coverage.

```bash
awk 'FNR==1 && NR>1 {next} {print}' \
    "$PAIR_OUT"/{gz10,desi_provabgs,galaxy_cruise,euclid_q1_morphology,galaxy_zoo_decals}.csv \
    > "$PAIR_OUT/all.csv"
wc -l "$PAIR_OUT/all.csv"  # 140 pairs across five catalogs, plus the header
awk -F, 'NR==1 || $5+$6>0' "$PAIR_OUT/all.csv" | column -t -s,
```
