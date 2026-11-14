# Rusty: snapshot and publish a larger LSST DP2 HATS release

The standalone [rusty_release.slurm](rusty_release.slurm) launcher builds from
existing coadds while the download campaigns continue. It does not contact
Rubin, read an RSP token, change the active checkout, or copy the FITS mirror.

## What gets included

The default sources are `lsst_dp2_raw` and extensions `ext01` through `ext04`
under `/mnt/ceph/users/polymathic/mronceray`. The source selection is unchanged:

```sql
i_cModelMag < 21
AND i_extendedness = 1
AND griz_model_extendedness >= 0.8
AND sersic_no_data_flag = 0
AND sersic_unknown_flag = 0
AND sersic_reff_major > 0.6
```

Each source manifest is read in one read-only SQLite transaction. The snapshot
stores its observed statuses, catalog hash, selection report, source paths,
UTC timestamp, and launcher hash. The merged catalog is written through Astropy
and checked for identical IDs and masks after reloading; the earlier
`i_psfFlux.mask` metadata failure is avoided. Duplicate IDs, overlapping
patch-band keys, and different selections fail before extraction.

Only patches with six terminal states (`complete` or `unavailable`) and at
least one complete image enter the build. Pending/failed/running patches remain
excluded; they are not converted into missing bands. Missing or still
initializing extension catalogs are skipped and logged.

**This is a point-in-time release.** Products downloaded after its snapshot
are retained in their raw campaigns, but will need a subsequent release name.
Rerunning the same name resumes the frozen snapshot and validated shards.

## Outputs and resource profile

Defaults:

- `lsst_dp2_release_20261114_inputs/`: immutable merged catalog and manifest.
- `lsst_dp2_release_20261114/`: frozen shard inputs, Parquet, logs and report.
- 64 CPU cores, 256 GiB RAM, 48-hour allocation, no GPU.
- 512 shards; 16 concurrent extractors with 4 patch threads each.
- HATS ingestion: 16 Dask workers and a 512-row partition threshold.
- Image contract and scalar photometry are the same as the main LSST README.
- Existing intermediate shards from a different release are not reused because
  their receipts are bound to another plan hash. The raw FITS are reused.

The previous 1,846,263-row build took 13:32:53 on Rusty with 32 requested cores
and 128 GiB. This is a measured baseline, not a finish-time guarantee for larger
snapshots. Queueing and Ceph throughput affect the new build. Keep several TiB
of free allocation for intermediate Parquet, HATS reduction and final output;
filesystem free space does not establish project quota.

## Fetch tools without changing the active checkout

Run on Rusty. The interpreter remains the existing MMU environment. The test
suite can validate the downloaded launcher against that exact checkout.

```bash
export MMU_REPO="$HOME/MultimodalUniverse"
export MMU_PYTHON="$MMU_REPO/.venv/bin/python"
export TOOLS="$HOME/mmu_jobs/lsst_release_20261114"
mkdir -p "$TOOLS"

git -C "$MMU_REPO" fetch origin feat/lsst-dp2-rusty-release
TOOLS_COMMIT=$(git -C "$MMU_REPO" rev-parse FETCH_HEAD)
git -C "$MMU_REPO" show "$TOOLS_COMMIT:scripts/lsst_dp2/rusty_release.slurm" > "$TOOLS/rusty_release.slurm"
git -C "$MMU_REPO" show "$TOOLS_COMMIT:tests/test_lsst_dp2_rusty_release.py" > "$TOOLS/test_lsst_dp2_rusty_release.py"
printf '%s\n' "$TOOLS_COMMIT" > "$TOOLS/tools_commit.txt"

export MMU_LSST_RELEASE_TEST_REPO="$MMU_REPO"
export MMU_LSST_RELEASE_TEST_SCRIPT="$TOOLS/rusty_release.slurm"
bash -n "$TOOLS/rusty_release.slurm"
"$MMU_PYTHON" -m unittest discover -s "$TOOLS" -p test_lsst_dp2_rusty_release.py -v
bash "$TOOLS/rusty_release.slurm" --check
```

Tests use small synthetic catalogs and directories. They check masked-column
round trips, frozen manifest states, source overlap, different selections,
publication guards, archiving and idempotence. They do not contact Rubin.

## Submit

Only submit if both the tests and `--check` pass. No new installation or
`uv sync` is needed, and no dependency on the network job should be added.

```bash
export LSST_DP2_RELEASE_NAME="lsst_dp2_release_20261114"
export LSST_DP2_MAX_EXTENSION=4
export LSST_DP2_AUTO_PUBLISH=1
export RUN="/mnt/ceph/users/polymathic/mronceray/$LSST_DP2_RELEASE_NAME"
mkdir -p "$RUN/logs"
JOB_ID=$(sbatch --parsable --export=ALL \
  --output="$RUN/logs/build-%j.out" \
  --error="$RUN/logs/build-%j.err" \
  "$TOOLS/rusty_release.slurm")
echo "$JOB_ID"
```

Set `LSST_DP2_AUTO_PUBLISH=0` to build without replacing the shared dataset.
The default in the launcher is zero.

## Validation and publication

The existing offline pipeline validates all intermediate images, their
checksums and receipts, all final IDs, and one final image per HATS partition.
The result is a technical consistency check, not independent scientific
calibration certification.

Publication requires `PASS`, matching counts and an expected HATS collection.
A candidate with fewer rows than the current shared release is rejected.
After validation, the whole collection (main catalog and margins) is moved to:

```text
/mnt/ceph/users/polymathic/MultimodalUniverse_v1.6/lsst_dp2/
```

The previous collection is retained under:

```text
/mnt/ceph/users/polymathic/mronceray/lsst_dp2_release_20261114_previous_published/
```

Both validation reports are relocated with their collection paths. The final
main catalog path remains:

```text
/mnt/ceph/users/polymathic/MultimodalUniverse_v1.6/lsst_dp2/lsst_dp2/
```

Schedule the publication as a release transition: the two directory renames
are not a continuous-reader transaction. No previous image data is deleted.

## Follow and recover

```bash
squeue -u "$USER" -n lsst-dp2-release
tail -n 20 --retry -F "$RUN/logs/build-$JOB_ID.out" "$RUN/logs/build-$JOB_ID.err"
find "$RUN/shards" -name validation.json | wc -l
sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

After publication, check the public `validation_report.json` and its
`hats_path`. The baseline release contains 1,846,263 rows (17.82% of the
10,358,863 global candidates matching these cuts). New row counts and coverage
must come from the new report, not from the full input-catalog count.

To resume a failed extraction/gather, rerun the same launcher with the same
release name and unchanged code/packages. Never delete the input plan to
refresh a release in place. To include later downloads, choose a new
`LSST_DP2_RELEASE_NAME`.
