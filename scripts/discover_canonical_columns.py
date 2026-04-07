"""Walk every built HATS catalog and dump its column schema as JSON.

Used to design the canonical-field registry. Run on the cluster after the
fiducial healpix catalogs have been built.
"""

import json
import os
import sys

import pyarrow.parquet as pq

DEFAULT_ROOT = "/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats"


def find_one_parquet(catalog_dir):
    for root, _, files in os.walk(os.path.join(catalog_dir, "dataset")):
        for f in files:
            if f.endswith(".parquet") and not f.startswith("_"):
                return os.path.join(root, f)
    return None


def schema_to_dict(schema):
    out = {}
    for f in schema:
        if f.name.startswith("_") or f.name in ("Norder", "Dir"):
            continue
        out[f.name] = str(f.type)
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    result = {}
    for ds_name in sorted(os.listdir(root)):
        ds_dir = os.path.join(root, ds_name)
        if not os.path.isdir(ds_dir) or ds_name in ("visualizations", "verification", "tmp"):
            continue
        for collection_name in sorted(os.listdir(ds_dir)):
            inner = os.path.join(ds_dir, collection_name, collection_name)
            if not os.path.isdir(inner):
                continue
            pf = find_one_parquet(inner)
            if pf is None:
                continue
            table = pq.read_metadata(pf).schema.to_arrow_schema()
            result[collection_name] = schema_to_dict(table)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
