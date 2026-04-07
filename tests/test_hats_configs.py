"""Tests for the per-dataset HATS config registry."""

import pytest

from mmu.hats_configs import (
    DATASET_CONFIGS,
    get_dataset_config,
    hdf5_path,
    list_datasets,
)


EXPECTED_DATASETS = {
    "sdss", "desi", "vipers", "galah", "apogee", "chandra",
    "gaia", "desi_provabgs",
    "legacysurvey", "ssl_legacysurvey", "hsc", "jwst", "btsbot", "gz10",
    "plasticc", "tess", "kepler", "foundation", "snls",
    "ps1_sne_ia", "des_y3_sne_ia", "swift_sne_ia", "yse",
    "allwise", "twomass", "galex", "sages",
}

SKIPPED_DATASETS = {"manga", "lamost", "cfa", "csp"}


class TestRegistry:
    def test_all_expected_datasets_present(self):
        non_skipped = {n for n, c in DATASET_CONFIGS.items() if not c.get("skip")}
        assert non_skipped == EXPECTED_DATASETS

    def test_manga_is_skipped(self):
        assert DATASET_CONFIGS["manga"].get("skip") is True

    def test_all_skipped_datasets_present(self):
        skipped = {n for n, c in DATASET_CONFIGS.items() if c.get("skip")}
        assert skipped == SKIPPED_DATASETS

    def test_every_entry_has_modality(self):
        for name, cfg in DATASET_CONFIGS.items():
            assert "modality" in cfg, f"{name} missing modality"

    def test_modalities_are_known(self):
        valid = {"spectra", "image", "timeseries", "tabular", "ifu"}
        for name, cfg in DATASET_CONFIGS.items():
            assert cfg["modality"] in valid, f"{name} has unknown modality {cfg['modality']}"

    def test_non_skipped_have_configs_and_default(self):
        for name, cfg in DATASET_CONFIGS.items():
            if cfg.get("skip"):
                continue
            assert "configs" in cfg, f"{name} missing configs"
            if cfg.get("flat_layout"):
                # Flat layout datasets (e.g. kepler) don't need a sub-config
                continue
            assert "default_config" in cfg, f"{name} missing default_config"
            assert cfg["default_config"] in cfg["configs"], (
                f"{name} default_config not in configs"
            )


class TestHelpers:
    def test_hdf5_path_format(self):
        path = hdf5_path("sdss", 583, config="sdss")
        assert path.endswith("/sdss/sdss/healpix=583/001-of-001.hdf5")
        assert path.startswith("/mnt/ceph/users/polymathic/MultimodalUniverse/")

    def test_hdf5_path_default_config(self):
        path = hdf5_path("sdss", 583)
        assert "/sdss/sdss/healpix=583/" in path  # uses default_config

    def test_hdf5_path_flat_layout(self):
        path = hdf5_path("kepler", 909)
        assert path.endswith("/kepler/healpix=909/001-of-001.hdf5")
        assert "/data/" not in path

    def test_get_dataset_config_unknown(self):
        with pytest.raises(KeyError):
            get_dataset_config("not_a_dataset")

    def test_get_dataset_config_skipped(self):
        with pytest.raises(ValueError, match="skipped"):
            get_dataset_config("manga")

    def test_get_dataset_config_returns_dict(self):
        cfg = get_dataset_config("sdss")
        assert cfg["modality"] == "spectra"
        assert "sdss" in cfg["configs"]


class TestListDatasets:
    def test_list_all(self):
        all_ds = list_datasets()
        assert "sdss" in all_ds
        assert "manga" not in all_ds
        assert len(all_ds) == len(EXPECTED_DATASETS)

    def test_list_by_modality(self):
        spectra = list_datasets("spectra")
        assert "sdss" in spectra
        assert "desi" in spectra
        assert "legacysurvey" not in spectra

        images = list_datasets("image")
        assert "legacysurvey" in images
        assert "hsc" in images
        assert "sdss" not in images

        timeseries = list_datasets("timeseries")
        assert "plasticc" in timeseries
        assert "kepler" in timeseries

        tabular = list_datasets("tabular")
        assert "allwise" in tabular
        assert "twomass" in tabular
