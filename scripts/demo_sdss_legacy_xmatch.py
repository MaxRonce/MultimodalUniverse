"""End-to-end demo: HATS-ify SDSS spectra and LegacySurvey images for one healpix tile,
cross-match them, and plot matched galaxies (image + spectrum side by side).

Run on the Flatiron cluster where MMU v1 HDF5 data is mounted at
/mnt/ceph/users/polymathic/MultimodalUniverse/.
"""

import argparse
import os

import h5py
import numpy as np
import pyarrow as pa
from astropy.table import Table as AstropyTable

from mmu.hats_import import build_arrow_table, np_to_pyarrow_list, write_hats

MMU_ROOT = "/mnt/ceph/users/polymathic/MultimodalUniverse"

SDSS_FLOAT = ["VDISP", "VDISP_ERR", "Z", "Z_ERR"]
SDSS_BOOL = ["ZWARNING"]
SDSS_FLUX = ["SPECTROFLUX", "SPECTROFLUX_IVAR", "SPECTROSYNFLUX", "SPECTROSYNFLUX_IVAR"]
SDSS_FILTERS = ["U", "G", "R", "I", "Z"]

LEGACY_FLOAT = [
    "FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z",
    "FIBERFLUX_G", "FIBERFLUX_R", "FIBERFLUX_I", "FIBERFLUX_Z",
    "EBV",
]


def load_sdss_tile(healpix: int) -> pa.Table:
    path = f"{MMU_ROOT}/sdss/sdss/healpix={healpix}/001-of-001.hdf5"
    print(f"Loading SDSS tile: {path}")
    with h5py.File(path, "r") as f:
        cols = [
            "ra", "dec", "object_id",
            "spectrum_flux", "spectrum_ivar", "spectrum_lambda",
            "spectrum_lsf_sigma", "spectrum_mask",
        ] + SDSS_FLOAT + SDSS_BOOL + SDSS_FLUX
        data = AstropyTable({k: f[k][:] for k in cols})
    table = build_arrow_table(
        data,
        float_features=SDSS_FLOAT,
        bool_features=SDSS_BOOL,
        flux_features=SDSS_FLUX,
        flux_filters=SDSS_FILTERS,
    )
    print(f"  SDSS: {table.num_rows} objects with spectra")
    return table


def load_legacy_tile_lazy(healpix: int, max_rows: int) -> pa.Table:
    """Read only the first `max_rows` from a LegacySurvey HDF5 tile, including image cutouts."""
    path = f"{MMU_ROOT}/legacysurvey/dr10_south_21/healpix={healpix}/001-of-001.hdf5"
    print(f"Loading LegacySurvey tile (first {max_rows} rows): {path}")
    with h5py.File(path, "r") as f:
        n = min(max_rows, f["object_id"].shape[0])

        columns = {
            "ra": pa.array(f["ra"][:n].astype(np.float64)),
            "dec": pa.array(f["dec"][:n].astype(np.float64)),
            "object_id": pa.array([str(o.decode() if isinstance(o, bytes) else o)
                                    for o in f["object_id"][:n]]),
        }

        for feat in LEGACY_FLOAT:
            if feat in f:
                columns[feat] = pa.array(f[feat][:n].astype(np.float32))

        # Read image cutouts as nested list-of-list (band, height, width)
        # Shape: (n, 4, 160, 160) — flatten last two dims to a list per band
        if "image_array" in f:
            imgs = f["image_array"][:n].astype(np.float32)
            n_obj, n_band, h, w = imgs.shape
            # Build a struct with one list-of-floats per row (band, h, w flattened)
            # Simpler: store as 1D list and remember shape
            flat = imgs.reshape(n_obj, -1)
            columns["image_flat"] = np_to_pyarrow_list(flat)
            columns["image_shape"] = pa.array(
                [[n_band, h, w]] * n_obj,
                type=pa.list_(pa.int32()),
            )

    table = pa.table(columns)
    print(f"  LegacySurvey: {table.num_rows} objects with image cutouts")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healpix", type=int, default=2300,
                        help="Healpix tile to process")
    parser.add_argument("--legacy-rows", type=int, default=2000,
                        help="Max LegacySurvey rows to read (limits memory)")
    parser.add_argument("--output", type=str,
                        default=os.path.expanduser("~/ceph/general_data/mmu_hats_demo_out"),
                        help="Output directory for HATS catalogs and plot")
    parser.add_argument("--radius-arcsec", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    sdss_dir = os.path.join(args.output, "sdss_demo", "sdss_demo")
    legacy_dir = os.path.join(args.output, "legacy_demo", "legacy_demo")

    # Build SDSS HATS catalog (skip if already built)
    if not os.path.exists(sdss_dir):
        sdss_table = load_sdss_tile(args.healpix)
        write_hats([sdss_table], args.output, "sdss_demo", debug=True)
    else:
        print(f"Reusing existing SDSS catalog at {sdss_dir}")

    # Build LegacySurvey HATS catalog (skip if already built)
    if not os.path.exists(legacy_dir):
        legacy_table = load_legacy_tile_lazy(args.healpix, args.legacy_rows)
        write_hats([legacy_table], args.output, "legacy_demo", debug=True)
    else:
        print(f"Reusing existing LegacySurvey catalog at {legacy_dir}")

    # Cross-match using LSDB
    from mmu.data import HATSDataset

    sdss_ds = HATSDataset(
        os.path.join(args.output, "sdss_demo", "sdss_demo"),
        columns=["ra", "dec", "Z", "object_id", "spectrum"],
    )
    legacy_ds = HATSDataset(
        os.path.join(args.output, "legacy_demo", "legacy_demo"),
        columns=["ra", "dec", "object_id", "image_flat", "image_shape",
                 "FLUX_G", "FLUX_R", "FLUX_Z"],
    )

    print(f"\nCross-matching SDSS x LegacySurvey at radius={args.radius_arcsec} arcsec...")
    matched = sdss_ds.crossmatch(
        legacy_ds,
        radius_arcsec=args.radius_arcsec,
        suffixes=("_sdss", "_legacy"),
    )
    print(f"Matched {matched.matched_count} pairs")

    # Visualize a few matched galaxies: image + spectrum
    if matched.matched_count > 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_show = min(4, matched.matched_count)
        fig, axes = plt.subplots(n_show, 2, figsize=(12, 3 * n_show))
        if n_show == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_show):
            item = matched[i]

            # Image: reconstruct from flat list + shape
            img_flat = np.array(item["image_flat_legacy"])
            shape = list(item["image_shape_legacy"])
            img = img_flat.reshape(shape)  # (4, 160, 160) — bands g, r, i, z
            # Use g, r, z bands for RGB-ish (indices 0, 1, 3)
            rgb = np.stack([img[3], img[1], img[0]], axis=-1)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            rgb = np.clip(rgb ** 0.5, 0, 1)
            axes[i, 0].imshow(rgb, origin="lower")
            axes[i, 0].set_title(
                f"LegacySurvey  ra={item['ra_legacy']:.4f}, dec={item['dec_legacy']:.4f}"
            )
            axes[i, 0].axis("off")

            # Spectrum
            spec = item["spectrum_sdss"]
            lam = np.array(spec["lambda"])
            flux = np.array(spec["flux"])
            mask = np.array(spec["mask"])
            flux_masked = np.where(mask, np.nan, flux)
            axes[i, 1].plot(lam, flux_masked, linewidth=0.5)
            axes[i, 1].set_title(
                f"SDSS spectrum  z={item['Z_sdss']:.4f}, sep={item['_dist_arcsec']:.2f}\""
            )
            axes[i, 1].set_xlabel("Wavelength [Å]")
            axes[i, 1].set_ylabel("Flux")
            axes[i, 1].set_xlim(3800, 9200)

        plt.tight_layout()
        plot_path = os.path.join(args.output, "xmatch_demo.png")
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        print(f"\nSaved visualization to {plot_path}")


if __name__ == "__main__":
    main()
