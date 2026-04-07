"""Per-dataset configuration for the HATS auto-converter.

Each dataset entry specifies:
- ``configs``: list of HDF5 sub-config directory names (e.g. SDSS has sdss/segue1/...)
- ``modality``: spectra/image/timeseries/tabular (for documentation/tests)
- ``invert_bool``: optional list of bool columns to invert
- ``image_columns``: optional override for the auto-image-detection list
- ``skip``: True if this dataset should not be auto-converted (e.g. manga)

The cluster path for an HDF5 tile is:
    /mnt/ceph/users/polymathic/MultimodalUniverse/{dataset}/{config}/healpix={N}/001-of-001.hdf5
"""

# Per-dataset config. Verified against /mnt/ceph/users/polymathic/MultimodalUniverse
# layout on the Flatiron cluster (snapshot taken 2026-04-07).
#
# ``flat_layout``: True for datasets that don't have a sub-config directory
# (healpix=N is directly under {dataset}/). E.g., kepler.

DATASET_CONFIGS = {
    # --- Spectra ---
    "sdss": {
        "configs": ["sdss", "boss", "eboss", "segue1", "segue2"],
        "modality": "spectra",
        "default_config": "sdss",
    },
    "desi": {
        "configs": ["dr1_main"],
        "modality": "spectra",
        "default_config": "dr1_main",
        "invert_bool": ["ZWARN"],
    },
    "vipers": {
        "configs": ["vipers_w1", "vipers_w4"],
        "modality": "spectra",
        "default_config": "vipers_w1",
    },
    "galah": {
        "configs": ["dr3"],
        "modality": "spectra",
        "default_config": "dr3",
    },
    "apogee": {
        "configs": ["apogee"],
        "modality": "spectra",
        "default_config": "apogee",
    },
    "chandra": {
        "configs": ["spectra"],
        "modality": "spectra",
        "default_config": "spectra",
    },
    "gaia": {
        "configs": ["gaia"],
        "modality": "spectra",  # spectral coefficients + nested structs
        "default_config": "gaia",
    },
    "desi_provabgs": {
        "configs": ["datafiles"],
        "modality": "tabular",  # MCMC posteriors
        "default_config": "datafiles",
    },

    # --- Images ---
    "legacysurvey": {
        "configs": ["dr10_south_21"],
        "modality": "image",
        "default_config": "dr10_south_21",
        "image_columns": ("image_array", "image_blobmodel", "image_rgb", "image_mask"),
    },
    "ssl_legacysurvey": {
        "configs": ["north"],  # cluster only has 'north', not stein_et_al
        "modality": "image",
        "default_config": "north",
    },
    "hsc": {
        "configs": ["pdr3_dud_22.5"],
        "modality": "image",
        "default_config": "pdr3_dud_22.5",
    },
    "jwst": {
        "configs": ["primer-cosmos", "primer-uds", "ceers", "ngdeep", "gds", "gdn"],
        "modality": "image",
        "default_config": "primer-cosmos",
    },
    "btsbot": {
        "configs": ["data"],  # cluster has 'data', not train/val/test
        "modality": "image",
        "default_config": "data",
    },
    "gz10": {
        "configs": ["datafiles"],  # cluster has 'datafiles' (HF builder uses gz10/gz10_rgb_images)
        "modality": "image",
        "default_config": "datafiles",
    },

    # --- Time series ---
    "plasticc": {
        "configs": ["data"],
        "modality": "timeseries",
        "default_config": "data",
    },
    "tess": {
        "configs": ["spoc"],  # only spoc on cluster (no qlp/tglc)
        "modality": "timeseries",
        "default_config": "spoc",
    },
    "kepler": {
        "configs": [],
        "modality": "timeseries",
        "flat_layout": True,  # healpix=N directly under kepler/
    },
    "foundation": {
        "configs": ["foundation_dr1"],
        "modality": "timeseries",
        "default_config": "foundation_dr1",
    },
    "snls": {
        "configs": ["data"],
        "modality": "timeseries",
        "default_config": "data",
    },
    "ps1_sne_ia": {
        "configs": ["ps1_sne_ia"],
        "modality": "timeseries",
        "default_config": "ps1_sne_ia",
    },
    "des_y3_sne_ia": {
        "configs": ["des_y3_sne_ia"],
        "modality": "timeseries",
        "default_config": "des_y3_sne_ia",
    },
    "swift_sne_ia": {
        "configs": ["data"],
        "modality": "timeseries",
        "default_config": "data",
    },
    "yse": {
        "configs": ["yse_dr1"],
        "modality": "timeseries",
        "default_config": "yse_dr1",
    },

    # --- Tabular ---
    "allwise": {
        "configs": ["allwise"],
        "modality": "tabular",
        "default_config": "allwise",
    },
    "twomass": {
        "configs": ["psc"],
        "modality": "tabular",
        "default_config": "psc",
    },
    "galex": {
        "configs": ["ais"],
        "modality": "tabular",
        "default_config": "ais",
    },
    "sages": {
        "configs": ["dr1"],
        "modality": "tabular",
        "default_config": "dr1",
    },

    # --- Skipped or missing on cluster ---
    "manga": {"skip": True, "modality": "ifu",
              "skip_reason": "IFU datacubes too different from rest of MMU"},
    "lamost": {"skip": True, "modality": "spectra",
               "skip_reason": "not present on cluster as of 2026-04-07"},
    "cfa": {"skip": True, "modality": "timeseries",
            "skip_reason": "not present on cluster as of 2026-04-07"},
    "csp": {"skip": True, "modality": "timeseries",
            "skip_reason": "not present on cluster as of 2026-04-07"},
}


MMU_V1_ROOT = "/mnt/ceph/users/polymathic/MultimodalUniverse"
MMU_V2_HATS_ROOT = "/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats"


def hdf5_path(dataset: str, healpix: int, config: str | None = None) -> str:
    """Return the path to an MMU v1 HDF5 tile on the cluster.

    For flat-layout datasets (like kepler), pass config=None.
    Otherwise, config is required.
    """
    cfg = DATASET_CONFIGS.get(dataset, {})
    if cfg.get("flat_layout"):
        return f"{MMU_V1_ROOT}/{dataset}/healpix={healpix}/001-of-001.hdf5"
    if config is None:
        config = cfg.get("default_config")
        if config is None:
            raise ValueError(f"config required for non-flat dataset {dataset}")
    return f"{MMU_V1_ROOT}/{dataset}/{config}/healpix={healpix}/001-of-001.hdf5"


def get_dataset_config(dataset: str) -> dict:
    """Return the config dict for a dataset, raising if missing or skipped."""
    if dataset not in DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset: {dataset}. Known: {sorted(DATASET_CONFIGS)}")
    cfg = DATASET_CONFIGS[dataset]
    if cfg.get("skip"):
        raise ValueError(f"Dataset {dataset} is marked as skipped: {cfg}")
    return cfg


def list_datasets(modality: str | None = None) -> list[str]:
    """List all non-skipped datasets, optionally filtered by modality."""
    return sorted(
        name for name, cfg in DATASET_CONFIGS.items()
        if not cfg.get("skip") and (modality is None or cfg["modality"] == modality)
    )
