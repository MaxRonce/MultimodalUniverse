"""Print spectrum_* keys for several MMU datasets at a given healpix tile."""

import sys
import h5py

DATASETS = [
    ("chandra", "spectra"),
    ("galah", "dr3"),
    ("desi", "dr1_main"),
    ("apogee", "apogee"),
    ("sdss", "sdss"),
]

healpix = int(sys.argv[1]) if len(sys.argv) > 1 else 1177

for ds, cfg in DATASETS:
    path = f"/mnt/ceph/users/polymathic/MultimodalUniverse/{ds}/{cfg}/healpix={healpix}/001-of-001.hdf5"
    try:
        with h5py.File(path, "r") as f:
            spec = sorted(k for k in f.keys() if k.startswith("spectrum"))
            print(f"{ds}/{cfg}: {spec}")
    except (OSError, FileNotFoundError) as e:
        print(f"{ds}/{cfg}: ERROR {e}")
