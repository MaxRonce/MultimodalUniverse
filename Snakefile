"""Snakemake pipeline: raw survey data -> HATS catalogs.

Each dataset has a `build_parent_sample_hats.py` under `scripts/{dataset}/`
that reads the survey-native raw inputs (parquet, FITS, etc.) and writes a
HATS catalog under `HATS_ROOT/{dataset}/{dataset}/{dataset}/`.

The Snakefile is intentionally a thin wrapper: it just declares one rule per
ported dataset and a top-level `all` target. Adding a new dataset is two lines
in `PORTED` plus the build script.

REQUIREMENTS: this pipeline only runs on the Flatiron cluster. Raw inputs live
under `/mnt/ceph/users/polymathic/external_data/astro/` and are NOT downloaded
by anything here. For local development use the unit tests in `tests/`.

Usage (run from a cluster login/compute node):

    # Build everything that's been ported, full size:
    uv run snakemake --cores 4 all

    # Build just one dataset:
    uv run snakemake --cores 1 build_allwise

    # Smoke-test against one raw shard per dataset:
    uv run snakemake --cores 1 build_allwise --config profile=test
"""

import os

from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT


configfile: "snakemake_config.yaml"

# Optional per-machine override (gitignored). See snakemake_config.local.yaml.example.
if os.path.exists("snakemake_config.local.yaml"):
    configfile: "snakemake_config.local.yaml"


def _resolve(key, default=None):
    """Look up `key` in the active profile, falling back to top-level config."""
    profile_name = config.get("profile", "cluster")
    profile = config.get("profiles", {}).get(profile_name, {})
    if key in config:
        return config[key]  # explicit --config wins
    return profile.get(key, default)


HATS_ROOT = _resolve("hats_root", MMU_V2_HATS_ROOT)
MAX_FILES = _resolve("max_files", None)  # None = build everything
RA_CENTER = _resolve("ra_center", None)
DEC_CENTER = _resolve("dec_center", None)
RADIUS = _resolve("radius", None)


# --------------------------------------------------------------------------- #
#                               OUTPUT SAFETY                                 #
# --------------------------------------------------------------------------- #
# /mnt/ceph/users/polymathic/ contains datasets that took MONTHS to produce,
# INCLUDING the READ-ONLY raw data mirror under external_data/ that every
# build script reads from. A Snakefile misconfiguration (typo in hats_root,
# wrong profile, stale envvar) could in principle ask Snakemake to write into
# these paths directly. Before ANY rule is parsed we validate HATS_ROOT so a
# bad value aborts the pipeline before a single directory is created.
#
# The validation is in `mmu.safety.validate_hats_root` so we can unit-test it
# rigorously. If you're adding a new allowed output prefix, update the
# whitelist there (and the tests).
from mmu.safety import validate_hats_root

validate_hats_root(HATS_ROOT)

# Datasets that have a working raw -> HATS build script under scripts/{name}/
PORTED = [
    "sdss",      # raw FITS plate files
    "allwise",   # raw IRSA parquet, healpix-prefiltered for cone cuts
    "twomass",   # raw IRSA gzipped CSV
    "sages",     # single FITS table (DR1 u/v photometry)
    "galex",     # multi-FITS GUVCat shards (latitude-partitioned)
    "desi",      # raw DESI coadd FITS files, merged via desispec.coadd_cameras
    "tess",      # raw TESS-SPOC FFI lightcurves (one file per TIC, sector)
    "ssl_legacysurvey",  # raw Stein-et-al DECaLS image chunks (h5 per 1M objects)
    "gaia",      # raw Gaia DR3 (GaiaSource + XpContinuousMeanSpectrum joined on source_id)
    "legacysurvey",  # raw DECaLS DR10 south sweeps + brick coadds (image cutouts + nearby catalog)
    "manga",     # raw SDSS-IV MaNGA IFU LOGCUBE + DAP MAPS files (spaxels + griz images + analysis maps)
]


def catalog_marker(name: str) -> str:
    """Path to the inner-catalog `hats.properties` file (the per-dataset target)."""
    return f"{HATS_ROOT}/{name}/{name}/{name}/hats.properties"


# Datasets where the script's filenames don't encode RA/Dec, so a cone cut
# would require opening every one of O(100k) files just to read headers.
# For these we fall back to a ``--max-files`` cap when a cone is requested,
# which gives a usable test slice without ~hours of header I/O. A proper
# production build of these datasets simply doesn't set cone args.
DATASETS_WITHOUT_CONE_SUPPORT = {"tess"}
CONE_FALLBACK_MAX_FILES = 3


def build_command(dataset: str) -> str:
    parts = [
        "python", "-m", f"scripts.{dataset}.build_parent_sample_hats",
        "--output-root", f"{HATS_ROOT}/{dataset}",
    ]

    cone_active = (
        RA_CENTER is not None and DEC_CENTER is not None and RADIUS is not None
    )

    if dataset in DATASETS_WITHOUT_CONE_SUPPORT and cone_active:
        # Cap the file count so test-slice builds of cone-incompatible
        # datasets finish quickly. The script will still write a valid HATS
        # catalog, just against the first few input files rather than a
        # sky-region slice.
        effective_max = MAX_FILES if MAX_FILES is not None else CONE_FALLBACK_MAX_FILES
        parts += ["--max-files", str(effective_max)]
        return " ".join(parts)

    if MAX_FILES is not None:
        parts += ["--max-files", str(MAX_FILES)]
    if cone_active:
        parts += [
            "--ra-center", str(RA_CENTER),
            "--dec-center", str(DEC_CENTER),
            "--radius", str(RADIUS),
        ]
    return " ".join(parts)


rule all:
    input:
        [catalog_marker(name) for name in PORTED],


rule build_allwise:
    output:
        marker = catalog_marker("allwise"),
    params:
        cmd = build_command("allwise"),
    shell:
        "{params.cmd}"


rule build_sdss:
    output:
        marker = catalog_marker("sdss"),
    params:
        cmd = build_command("sdss"),
    shell:
        "{params.cmd}"


rule build_twomass:
    output:
        marker = catalog_marker("twomass"),
    params:
        cmd = build_command("twomass"),
    shell:
        "{params.cmd}"


rule build_sages:
    output:
        marker = catalog_marker("sages"),
    params:
        cmd = build_command("sages"),
    shell:
        "{params.cmd}"


rule build_galex:
    output:
        marker = catalog_marker("galex"),
    params:
        cmd = build_command("galex"),
    shell:
        "{params.cmd}"


rule build_desi:
    output:
        marker = catalog_marker("desi"),
    params:
        cmd = build_command("desi"),
    shell:
        "{params.cmd}"


rule build_tess:
    output:
        marker = catalog_marker("tess"),
    params:
        cmd = build_command("tess"),
    shell:
        "{params.cmd}"


rule build_ssl_legacysurvey:
    output:
        marker = catalog_marker("ssl_legacysurvey"),
    params:
        cmd = build_command("ssl_legacysurvey"),
    shell:
        "{params.cmd}"


rule build_gaia:
    output:
        marker = catalog_marker("gaia"),
    params:
        cmd = build_command("gaia"),
    shell:
        "{params.cmd}"


rule build_legacysurvey:
    output:
        marker = catalog_marker("legacysurvey"),
    params:
        cmd = build_command("legacysurvey"),
    shell:
        "{params.cmd}"


rule build_manga:
    output:
        marker = catalog_marker("manga"),
    params:
        cmd = build_command("manga"),
    shell:
        "{params.cmd}"
