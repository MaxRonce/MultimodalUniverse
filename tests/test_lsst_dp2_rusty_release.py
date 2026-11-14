"""Lightweight tests of the Python stages embedded in the Rusty launcher.

These use synthetic catalogs and directory fixtures, never Rubin or image data.
Only independent filesystem helpers are loaded from the production modules.
"""
import ast
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.table import MaskedColumn, Table

REPO = Path(os.environ.get(
    "MMU_LSST_RELEASE_TEST_REPO", str(Path(__file__).resolve().parents[1])
)).resolve()
SCRIPT = Path(os.environ.get(
    "MMU_LSST_RELEASE_TEST_SCRIPT", str(REPO / "scripts/lsst_dp2/rusty_release.slurm")
)).resolve()


def python_stages():
    blocks, current = [], None
    for line in SCRIPT.read_text().splitlines():
        if current is not None:
            if line == "PY":
                blocks.append("\n".join(current))
                current = None
            else:
                current.append(line)
        elif "<<'PY'" in line:
            current = []
    assert current is None and len(blocks) == 3
    for block in blocks:
        ast.parse(block)
    return blocks


PREFLIGHT, SNAPSHOT, PUBLISH = python_stages()


def independent_module(name, path, names):
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            selected.append(node)
    module = types.ModuleType(name)
    module.__dict__.update(
        os=os, Path=Path, sqlite3=sqlite3, hashlib=hashlib, defaultdict=defaultdict
    )
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


class RustyReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.run = self.base / "test_release"
        self.run.mkdir()
        self.snapshot = self.base / "test_release_inputs"
        self.live = self.base / "shared/lsst_dp2"
        self.archive = self.base / "test_release_previous_published"
        (self.base / "tmp").mkdir()
        self.env = {
            "DP2_BASE": str(self.base),
            "TMPDIR": str(self.base / "tmp"),
            "LSST_DP2_RELEASE_RUN": str(self.run),
            "LSST_DP2_RELEASE_SNAPSHOT": str(self.snapshot),
            "LSST_DP2_RELEASE_SCRIPT": str(SCRIPT),
            "LSST_DP2_RELEASE_NAME": "test_release",
            "LSST_DP2_MAX_EXTENSION": "4",
            "LSST_DP2_AUTO_PUBLISH": "1",
            "LSST_DP2_PUBLISH_ROOT": str(self.live),
        }
        environment = patch.dict(os.environ, self.env)
        environment.start()
        self.addCleanup(environment.stop)
        common = independent_module(
            "scripts.lsst_dp2.common",
            REPO / "scripts/lsst_dp2/common.py",
            {"BANDS", "sha256_file"},
        )
        helpers = independent_module(
            "scripts.lsst_dp2.offline_cutouts",
            REPO / "scripts/lsst_dp2/offline_cutouts.py",
            {"MANIFEST_COLUMNS", "_write_manifest", "ready_patches"},
        )
        helpers.BANDS = common.BANDS
        self.helpers = helpers
        scripts = types.ModuleType("scripts")
        scripts.__path__ = []
        package = types.ModuleType("scripts.lsst_dp2")
        package.__path__ = []
        scripts.lsst_dp2 = package
        package.common, package.offline_cutouts = common, helpers
        modules = patch.dict("sys.modules", {
            "scripts": scripts, "scripts.lsst_dp2": package,
            "scripts.lsst_dp2.common": common,
            "scripts.lsst_dp2.offline_cutouts": helpers,
        })
        modules.start()
        self.addCleanup(modules.stop)

    def execute(self, block):
        exec(compile(block, str(SCRIPT), "exec"), {"__name__": "__main__"})

    def source(self, name, ids, patch_id, masked=None, pending=False, where="i < 21"):
        source = self.base / name
        (source / "catalog").mkdir(parents=True)
        catalog = source / "catalog/objects.parquet"
        table = Table({"objectId": np.array(ids, dtype=np.int64)})
        table["i_psfFlux"] = MaskedColumn(
            np.arange(len(ids), dtype=float), mask=masked or [False] * len(ids)
        )
        table.write(catalog, format="parquet")
        Path(str(catalog) + ".selection.json").write_text(
            json.dumps({"status": "PASS", "where": where, "objects": len(ids)})
        )
        rows = [{
            "tract": 1, "patch": patch_id, "band": band,
            "output_path": str(source / f"{band}.fits"),
            "sha256": band * 64, "dataset_id": f"dataset-{patch_id}-{band}",
            "s_resolution": 1.0, "mask_planes": "BAD,SAT,NO_DATA",
            "status": "pending" if pending and band == "u" else "complete",
        } for band in "ugrizy"]
        self.helpers._write_manifest(source / "download_manifest.sqlite", rows)
        return source

    def publication(self, count, old_count=1, status="PASS"):
        collection = self.run / "hats/lsst_dp2/lsst_dp2"
        inner = collection / "lsst_dp2"
        (inner / "dataset").mkdir(parents=True)
        (inner / "hats.properties").write_text(f"hats_nrows={count}\n")
        (inner / "dataset/new.parquet").write_bytes(b"new")
        (self.live / "lsst_dp2/dataset").mkdir(parents=True)
        (self.live / "lsst_dp2/hats.properties").write_text(f"hats_nrows={old_count}\n")
        (self.live / "lsst_dp2/dataset/old.parquet").write_bytes(b"old")
        old_report = {
            "status": "PASS", "plan_sha256": "old",
            "hats_rows": old_count, "catalog_rows": old_count,
            "unique_object_ids": old_count,
            "hats_path": str(self.live / "lsst_dp2"),
            "hats_properties": str(self.live / "lsst_dp2/hats.properties"),
        }
        (self.live / "validation_report.json").write_text(json.dumps(old_report))
        new_report = {
            "status": status, "plan_sha256": "new",
            "hats_rows": count, "catalog_rows": count, "unique_object_ids": count,
            "hats_path": str(inner), "hats_properties": str(inner / "hats.properties"),
        }
        (self.run / "validation_report.json").write_text(json.dumps(new_report))
        return collection

    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_masked_preflight(self):
        self.source("lsst_dp2_raw", [1], 1)
        self.execute(PREFLIGHT)

    def test_snapshot_preserves_masks_and_freezes_manifest_states(self):
        self.source("lsst_dp2_raw", [1, 2], 1, masked=[False, True])
        second = self.source("lsst_dp2_raw_ext01", [3], 2, pending=True)
        self.execute(SNAPSHOT)
        table = Table.read(self.snapshot / "catalog/objects.parquet", format="parquet")
        np.testing.assert_array_equal(table["objectId"], [1, 2, 3])
        np.testing.assert_array_equal(np.ma.getmaskarray(table["i_psfFlux"]), [False, True, False])
        with sqlite3.connect(second / "download_manifest.sqlite") as con:
            con.execute("UPDATE coadds SET status='complete'")
        self.execute(SNAPSHOT)
        with sqlite3.connect(self.snapshot / "download_manifest.sqlite") as con:
            con.row_factory = sqlite3.Row
            rows = [dict(r) for r in con.execute("SELECT * FROM coadds")]
        self.assertEqual(self.helpers.ready_patches(rows), {(1, 1)})
        metadata = json.loads((self.snapshot / "catalog/objects.parquet.selection.json").read_text())
        self.assertEqual(metadata["objects"], 3)
        self.assertEqual(len(metadata["sources"]), 2)

    def test_duplicate_identifiers_rejected(self):
        self.source("lsst_dp2_raw", [1], 1)
        self.source("lsst_dp2_raw_ext01", [1], 2)
        with self.assertRaisesRegex(AssertionError, "Duplicate objects"):
            self.execute(SNAPSHOT)
        self.assertFalse(self.snapshot.exists())

    def test_overlapping_patches_rejected(self):
        self.source("lsst_dp2_raw", [1], 1)
        self.source("lsst_dp2_raw_ext01", [2], 1)
        with self.assertRaisesRegex(AssertionError, "Overlapping"):
            self.execute(SNAPSHOT)
        self.assertFalse(self.snapshot.exists())

    def test_different_selections_rejected(self):
        self.source("lsst_dp2_raw", [1], 1)
        self.source("lsst_dp2_raw_ext01", [2], 2, where="i < 24")
        with self.assertRaisesRegex(AssertionError, "Different selections"):
            self.execute(SNAPSHOT)
        self.assertFalse(self.snapshot.exists())

    def test_publication_archives_previous_and_is_idempotent(self):
        collection = self.publication(2)
        self.execute(PUBLISH)
        self.assertFalse(collection.exists())
        self.assertEqual((self.live / "lsst_dp2/dataset/new.parquet").read_bytes(), b"new")
        self.assertEqual((self.archive / "lsst_dp2/dataset/old.parquet").read_bytes(), b"old")
        old = json.loads((self.archive / "validation_report.json").read_text())
        self.assertEqual(old["hats_path"], str(self.archive / "lsst_dp2"))
        report = json.loads((self.live / "validation_report.json").read_text())
        self.assertEqual(report["hats_rows"], 2)
        self.assertEqual(report["hats_path"], str(self.live / "lsst_dp2"))
        self.execute(PUBLISH)
        self.assertTrue((self.archive / "lsst_dp2/dataset/old.parquet").is_file())

    def test_failed_validation_does_not_replace_live_dataset(self):
        collection = self.publication(2, status="FAIL")
        with self.assertRaises(AssertionError):
            self.execute(PUBLISH)
        self.assertTrue(collection.exists())
        self.assertTrue((self.live / "lsst_dp2/dataset/old.parquet").is_file())
        self.assertFalse(self.archive.exists())

    def test_smaller_release_does_not_replace_live_dataset(self):
        self.publication(1, old_count=2)
        with self.assertRaisesRegex(AssertionError, "fewer rows"):
            self.execute(PUBLISH)
        self.assertTrue((self.live / "lsst_dp2/dataset/old.parquet").is_file())
        self.assertFalse(self.archive.exists())


if __name__ == "__main__":
    unittest.main()
