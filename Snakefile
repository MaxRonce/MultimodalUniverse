"""Snakemake pipeline for converting MMU v1 HDF5 datasets to HATS format.

Run from the repo root:

    # Run all conversions for the fiducial healpix tile (1 sq deg slice):
    uv run snakemake --cores 4 all_hats_fiducial

    # Run conversion for one (dataset, config, healpix):
    uv run snakemake --cores 1 \\
        /mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/sdss/sdss_sdss/sdss_sdss/hats.properties

    # Verify all built catalogs:
    uv run snakemake --cores 4 verify_all

    # Visualize all built catalogs:
    uv run snakemake --cores 4 visualize_all

The pipeline reads from MMU v1 at:
    /mnt/ceph/users/polymathic/MultimodalUniverse/{dataset}/{config}/healpix={pix}/...

and writes HATS output to:
    /mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/{dataset}/{dataset}_{config}/...
"""

import os
from mmu.hats_configs import DATASET_CONFIGS, MMU_V1_ROOT, MMU_V2_HATS_ROOT, hdf5_path

# ---- Configurable parameters ----
configfile: "snakemake_config.yaml"

FIDUCIAL_HEALPIX = config.get("fiducial_healpix", 1177)
N_ROWS = config.get("n_rows", 2000)
HATS_ROOT = config.get("hats_root", MMU_V2_HATS_ROOT)


# ---- Build the (dataset, config) targets that exist on disk ----
def discover_targets(healpix):
    """Yield (dataset, config) pairs that have HDF5 data at the given healpix tile."""
    for dataset, dscfg in DATASET_CONFIGS.items():
        if dscfg.get("skip"):
            continue
        if dscfg.get("flat_layout"):
            path = hdf5_path(dataset, healpix)
            if os.path.exists(path):
                yield (dataset, None)
            continue
        for sub in dscfg["configs"]:
            path = hdf5_path(dataset, healpix, config=sub)
            if os.path.exists(path):
                yield (dataset, sub)


def catalog_path(dataset, sub):
    name = f"{dataset}_{sub}" if sub else dataset
    return f"{HATS_ROOT}/{dataset}/{name}/{name}/hats.properties"


def visualization_path(dataset, sub):
    name = f"{dataset}_{sub}" if sub else dataset
    return f"{HATS_ROOT}/visualizations/{name}.png"


def verify_marker_path(dataset, sub):
    name = f"{dataset}_{sub}" if sub else dataset
    return f"{HATS_ROOT}/verification/{name}.ok"


# ---- Top-level rules ----

rule all_hats_fiducial:
    """Build HATS catalogs for every dataset that has data in the fiducial tile."""
    input:
        [catalog_path(ds, sub) for ds, sub in discover_targets(FIDUCIAL_HEALPIX)],


rule verify_all:
    input:
        [verify_marker_path(ds, sub) for ds, sub in discover_targets(FIDUCIAL_HEALPIX)],


rule visualize_all:
    input:
        [visualization_path(ds, sub) for ds, sub in discover_targets(FIDUCIAL_HEALPIX)],


# ---- Per-target rules ----

rule build_hats:
    """Build a HATS catalog for one (dataset, config) at the fiducial healpix tile."""
    output:
        marker = HATS_ROOT + "/{dataset}/{dataset}_{config}/{dataset}_{config}/hats.properties",
    params:
        out_root = lambda wc: f"{HATS_ROOT}/{wc.dataset}",
        n_rows = N_ROWS,
        healpix = FIDUCIAL_HEALPIX,
    shell:
        """
        python -m mmu.cli.build_hats \\
            --dataset {wildcards.dataset} \\
            --config {wildcards.config} \\
            --healpix {params.healpix} \\
            --n-rows {params.n_rows} \\
            --output {params.out_root}
        """


rule verify_hats:
    """Run the verification battery on a built HATS catalog."""
    input:
        marker = HATS_ROOT + "/{dataset}/{dataset}_{config}/{dataset}_{config}/hats.properties",
    output:
        ok = HATS_ROOT + "/verification/{dataset}_{config}.ok",
    params:
        catalog = HATS_ROOT + "/{dataset}/{dataset}_{config}/{dataset}_{config}",
        source = lambda wc: hdf5_path(wc.dataset, FIDUCIAL_HEALPIX, config=wc.config),
    shell:
        """
        mkdir -p $(dirname {output.ok})
        python -m mmu.verify --catalog {params.catalog} --source-hdf5 {params.source} > {output.ok}
        """


rule visualize_hats:
    """Generate a preview PNG for a built HATS catalog."""
    input:
        marker = HATS_ROOT + "/{dataset}/{dataset}_{config}/{dataset}_{config}/hats.properties",
    output:
        png = HATS_ROOT + "/visualizations/{dataset}_{config}.png",
    params:
        catalog = HATS_ROOT + "/{dataset}/{dataset}_{config}/{dataset}_{config}",
    shell:
        """
        mkdir -p $(dirname {output.png})
        python scripts/visualize_hats_dataset.py --catalog {params.catalog} --output {output.png}
        """
