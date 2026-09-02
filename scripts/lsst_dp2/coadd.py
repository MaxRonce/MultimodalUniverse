"""Read Rubin DP2 deep-coadd products and extract calibrated source stamps."""

from __future__ import annotations

import contextlib
import json

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from scripts.lsst_dp2.common import (
    DEFAULT_REJECT_MASK_PLANES,
    IMAGE_SIZE,
    mask_plane_mapping,
    native_array,
)

PSF_SIZE = 35
CORE_REQUIRED_MASK_PLANES = ("SAT", "NO_DATA")
MASK_PLANE_ALIASES = {
    "SAT": ("SAT", "SATURATED"),
    "INTRP": ("INTRP", "INTERPOLATED"),
    "CR": ("CR", "COSMIC_RAY"),
    "EDGE": ("EDGE", "DETECTION_EDGE"),
}


def _find_hdu(hdul: fits.HDUList, names: set[str]):
    for hdu in hdul:
        if str(hdu.header.get("EXTNAME", "")).upper() in names:
            return hdu
    raise ValueError(f"missing FITS extension in {sorted(names)}")


def _read_archive_json(hdul: fits.HDUList) -> dict:
    json_hdu = _find_hdu(hdul, {"JSON"})
    return json.loads(bytes(json_hdu.data["JSON"][0]))


def clean_mask_from_bits(
    mask_bits: np.ndarray,
    mapping: dict[str, int],
    reject_planes: tuple[str, ...] = DEFAULT_REJECT_MASK_PLANES,
) -> np.ndarray:
    """Return true for pixels that do not intersect a rejected Rubin plane."""

    def resolve(name: str) -> str | None:
        return next(
            (
                candidate
                for candidate in MASK_PLANE_ALIASES.get(name, (name,))
                if candidate in mapping
            ),
            None,
        )

    missing_core = [name for name in CORE_REQUIRED_MASK_PLANES if resolve(name) is None]
    if missing_core:
        raise ValueError(f"mask metadata lacks required planes: {missing_core}")

    clean = np.ones(mask_bits.shape, dtype=bool)
    for name in reject_planes:
        native_name = resolve(name)
        if native_name is not None:
            clean &= (mask_bits & (1 << mapping[native_name])) == 0
    return clean


@contextlib.contextmanager
def open_maskedimage(path: str):
    """Open one mirrored product while keeping its memory-mapped arrays alive."""
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        image_hdu = _find_hdu(hdul, {"IMAGE", "SCI", "SCIENCE"})
        mask_hdu = _find_hdu(hdul, {"MASK"})
        variance_hdu = _find_hdu(hdul, {"VARIANCE", "VAR"})
        psf_hdu = _find_hdu(hdul, {"PSF"})
        archive = _read_archive_json(hdul)
        grid = archive.get("psf", {}).get("bounds", {}).get("grid", {})
        grid_bbox = grid.get("bbox", {})
        cell_shape = tuple(grid.get("cell_shape", ()))
        image_yx0 = tuple(archive.get("image", {}).get("yx0", ()))
        psf_array = native_array(np.asarray(psf_hdu.data, dtype=np.float32))

        if psf_array.ndim != 4 or psf_array.shape[-2:] != (PSF_SIZE, PSF_SIZE):
            raise ValueError(f"unexpected cell PSF shape in {path}: {psf_array.shape}")
        if len(cell_shape) != 2 or len(image_yx0) != 2:
            raise ValueError(f"incomplete cell PSF metadata in {path}")

        yield {
            "image": image_hdu.data,
            "variance": variance_hdu.data,
            "mask_bits": mask_hdu.data,
            "wcs": WCS(image_hdu.header),
            "mask_mapping": mask_plane_mapping(
                hdul[0].header, image_hdu.header, mask_hdu.header
            ),
            "psf_array": psf_array,
            "psf_grid_start_yx": (
                int(grid_bbox["y"]["start"]),
                int(grid_bbox["x"]["start"]),
            ),
            "psf_cell_shape_yx": (int(cell_shape[0]), int(cell_shape[1])),
            "image_yx0": (int(image_yx0[0]), int(image_yx0[1])),
        }


def psf_kernel_at(coadd: dict, ra: float, dec: float) -> tuple[np.ndarray, bool]:
    """Return the normalized local cell PSF, or zero plus false when unavailable."""
    x, y = coadd["wcs"].world_to_pixel_values(ra, dec)
    absolute_y = y + coadd["image_yx0"][0]
    absolute_x = x + coadd["image_yx0"][1]
    grid_y0, grid_x0 = coadd["psf_grid_start_yx"]
    cell_y, cell_x = coadd["psf_cell_shape_yx"]
    iy = int(np.floor((absolute_y - grid_y0) / cell_y))
    ix = int(np.floor((absolute_x - grid_x0) / cell_x))
    psf_array = coadd["psf_array"]

    if not (0 <= iy < psf_array.shape[0] and 0 <= ix < psf_array.shape[1]):
        raise ValueError(f"source is outside PSF grid: cell={(iy, ix)}")

    kernel = np.asarray(psf_array[iy, ix], dtype=np.float32).copy()
    if not np.isfinite(kernel).all():
        return np.zeros((PSF_SIZE, PSF_SIZE), dtype=np.float32), False
    total = float(np.sum(kernel, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        return np.zeros((PSF_SIZE, PSF_SIZE), dtype=np.float32), False
    kernel /= total
    return kernel, True


def make_stamp(
    coadd: dict,
    ra: float,
    dec: float,
    reject_planes: tuple[str, ...] = DEFAULT_REJECT_MASK_PLANES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract calibrated flux, inverse variance, clean mask, and raw mask bits."""
    position = SkyCoord(ra=ra, dec=dec, unit="deg")
    shape = (IMAGE_SIZE, IMAGE_SIZE)
    cutout_args = {
        "position": position,
        "size": shape,
        "wcs": coadd["wcs"],
        "mode": "partial",
    }
    image = Cutout2D(coadd["image"], fill_value=np.nan, **cutout_args).data
    variance = Cutout2D(coadd["variance"], fill_value=np.nan, **cutout_args).data
    bits = Cutout2D(coadd["mask_bits"], fill_value=0, **cutout_args).data
    coverage = Cutout2D(
        np.ones(coadd["image"].shape, dtype=np.uint8), fill_value=0, **cutout_args
    ).data.astype(bool)

    bits = native_array(np.asarray(bits, dtype=np.int32))
    clean = coverage & clean_mask_from_bits(bits, coadd["mask_mapping"], reject_planes)
    finite = np.isfinite(image) & np.isfinite(variance) & (variance > 0)
    valid = clean & finite
    flux = np.nan_to_num(
        native_array(image).astype(np.float32),
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    ivar = np.zeros(shape, dtype=np.float32)
    np.divide(1.0, variance, out=ivar, where=valid)
    return flux, ivar, valid, bits
