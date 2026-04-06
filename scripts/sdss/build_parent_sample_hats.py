import os
import argparse
import numpy as np
from astropy.io import fits
from astropy.table import Table, join
from multiprocessing import Pool
from tqdm import tqdm
import healpy as hp
import pyarrow as pa
from dask.distributed import Client
from hats_import import CollectionArguments
from hats_import.catalog.file_readers import InputReader
from hats_import.pipeline import pipeline_with_client

_healpix_nside = 16

SURVEYS = ['sdss  ',
           'segue1',
           'segue2',
           'boss  ',
           'eboss ']

# Schema definition matching the SDSS transformer
FLOAT_FEATURES = ["VDISP", "VDISP_ERR", "Z", "Z_ERR"]
BOOL_FEATURES = ["ZWARNING"]
FLUX_FEATURES = ["SPECTROFLUX", "SPECTROFLUX_IVAR", "SPECTROSYNFLUX", "SPECTROSYNFLUX_IVAR"]
FLUX_FILTERS = ["U", "G", "R", "I", "Z"]


def np_to_pyarrow_list(array):
    """Convert a 2D numpy array to a PyArrow ListArray."""
    if array.dtype.byteorder == '>':
        array = array.byteswap().view(array.dtype.newbyteorder('<'))
    values = pa.array(array.reshape(-1))
    if array.ndim == 1:
        return values
    n_lists, length = array.shape
    offsets = np.arange(0, (n_lists + 1) * length, length, dtype=np.int32)
    return pa.ListArray.from_arrays(values=values, offsets=offsets)


def selection_fn(catalog):
    mask = catalog['SPECPRIMARY'] == 1
    mask &= catalog['TARGETTYPE'] == "SCIENCE "
    mask &= catalog['PLATEQUALITY'] == "good    "
    return mask


def processing_fn(args):
    """Parallel processing function reading all requested spectra from one plate."""
    filename, fiber_ids, object_id = args
    fiber_ids = fiber_ids - 1

    hdus = fits.open(filename)

    flux = hdus[0].data[fiber_ids]
    ivar = hdus[1].data[fiber_ids]
    and_mask = hdus[2].data[fiber_ids]
    lsf_sigma = hdus[4].data[fiber_ids]

    mask = and_mask.astype(bool) | (ivar <= 1e-6)

    loglam = hdus[0].header['CRVAL1'] + hdus[0].header['CD1_1'] * (np.arange(len(flux[0])) + 1 - hdus[0].header['CRPIX1'])
    lam = np.repeat(10**loglam.reshape(1, -1), len(fiber_ids), axis=0).astype(np.float32)

    return {'object_id': object_id,
            'spectrum_lambda': lam.astype(np.float32),
            'spectrum_flux': flux,
            'spectrum_ivar': ivar,
            'spectrum_mask': mask,
            'spectrum_lsf_sigma': lsf_sigma}


def process_healpix_group(args):
    """Process one healpix group and return a PyArrow table."""
    catalog, sdss_data_path = args

    catalog['ra'] = catalog['PLUG_RA']
    catalog['dec'] = catalog['PLUG_DEC']
    catalog['object_id'] = catalog['SPECOBJID']

    catalog = catalog.group_by(['SURVEY', 'PLATE'])

    map_args = []
    for group in catalog.groups:
        survey = group['SURVEY'][0]
        plate = group['PLATE'][0]
        mjd = group['MJD'][0]
        fiberid = group['FIBERID']
        object_id = group['object_id']
        filename = "spPlate-{}-{}.fits".format(str(plate).zfill(4), mjd)
        map_args += [(os.path.join(sdss_data_path, survey.strip(), str(plate).zfill(4), filename),
                      fiberid, object_id)]

    results = []
    for a in map_args:
        results.append(processing_fn(a))

    max_length = max([len(d['spectrum_flux'][0]) for d in results])
    for i in range(len(results)):
        results[i]['spectrum_flux'] = np.pad(results[i]['spectrum_flux'], ((0, 0), (0, max_length - len(results[i]['spectrum_flux'][0]))), mode='edge')
        results[i]['spectrum_ivar'] = np.pad(results[i]['spectrum_ivar'], ((0, 0), (0, max_length - len(results[i]['spectrum_ivar'][0]))), mode='constant')
        results[i]['spectrum_lambda'] = np.pad(results[i]['spectrum_lambda'], ((0, 0), (0, max_length - len(results[i]['spectrum_lambda'][0]))), mode='constant', constant_values=-1)
        results[i]['spectrum_lsf_sigma'] = np.pad(results[i]['spectrum_lsf_sigma'], ((0, 0), (0, max_length - len(results[i]['spectrum_lsf_sigma'][0]))), mode='edge')
        results[i]['spectrum_mask'] = np.pad(results[i]['spectrum_mask'], ((0, 0), (0, max_length - len(results[i]['spectrum_mask'][0]))), mode='constant', constant_values=True)

    spectra = Table({k: np.concatenate([d[k] for d in results], axis=0)
                     for k in results[0].keys()})

    catalog = join(catalog, spectra, keys='object_id', join_type='inner')
    assert len(catalog) == len(spectra), "Join error: some spectra files may be missing"

    return catalog_to_arrow(catalog)


def catalog_to_arrow(catalog):
    """Convert an astropy Table (with spectra) to a PyArrow table matching the HATS schema."""
    columns = {}

    # Spectrum struct
    spectrum_arrays = [
        np_to_pyarrow_list(np.array(catalog['spectrum_flux']).astype(np.float32)),
        np_to_pyarrow_list(np.array(catalog['spectrum_ivar']).astype(np.float32)),
        np_to_pyarrow_list(np.array(catalog['spectrum_lsf_sigma']).astype(np.float32)),
        np_to_pyarrow_list(np.array(catalog['spectrum_lambda']).astype(np.float32)),
        np_to_pyarrow_list(np.array(catalog['spectrum_mask'])),
    ]
    columns["spectrum"] = pa.StructArray.from_arrays(
        spectrum_arrays, names=["flux", "ivar", "lsf_sigma", "lambda", "mask"]
    )

    for f in FLOAT_FEATURES:
        columns[f] = pa.array(np.array(catalog[f]).astype(np.float32))

    columns["ra"] = pa.array(np.array(catalog['ra']).astype(np.float64))
    columns["dec"] = pa.array(np.array(catalog['dec']).astype(np.float64))

    for f in BOOL_FEATURES:
        columns[f] = pa.array(np.array(catalog[f]).astype(bool))

    for f in FLUX_FEATURES:
        flux_data = np.array(catalog[f])
        for n, b in enumerate(FLUX_FILTERS):
            columns[f"{f}_{b}"] = pa.array(flux_data[:, n].astype(np.float32))

    columns["object_id"] = pa.array([str(oid) for oid in catalog['object_id']])

    return pa.table(columns)


class ArrowTableReader(InputReader):
    """InputReader that yields pre-built PyArrow tables."""

    def __init__(self, tables):
        self.tables = tables

    def read(self, input_file, read_columns=None):
        idx = int(input_file)
        table = self.tables[idx]
        if read_columns:
            table = table.select(read_columns)
        yield table


def main(args):
    catalog = Table.read(os.path.join(args.sdss_data_path, "specObj-dr17.fits"))
    catalog = catalog[selection_fn(catalog)]
    catalog['healpix'] = hp.ang2pix(_healpix_nside, catalog['PLUG_RA'], catalog['PLUG_DEC'], lonlat=True, nest=True)

    for survey in SURVEYS:
        print("Processing survey:", survey)

        cat_survey = catalog[catalog['SURVEY'] == survey]
        if len(cat_survey) == 0:
            continue
        cat_survey = cat_survey.group_by(['healpix'])

        # Process each healpix group and collect PyArrow tables
        map_args = [(group, args.sdss_data_path) for group in cat_survey.groups]

        tables = []
        with Pool(args.num_processes) as pool:
            for table in tqdm(pool.imap(process_healpix_group, map_args), total=len(map_args)):
                tables.append(table)

        if not tables:
            continue

        survey_name = survey.strip()
        print(f"Writing HATS catalog for {survey_name} ({sum(t.num_rows for t in tables)} objects)...")

        reader = ArrowTableReader(tables)
        import_args = (
            CollectionArguments(
                output_artifact_name=f"sdss_{survey_name}",
                output_path=args.output_dir,
                tmp_dir=os.path.join(args.output_dir, "tmp"),
            )
            .catalog(
                input_file_list=[str(i) for i in range(len(tables))],
                file_reader=reader,
                ra_column="ra",
                dec_column="dec",
                pixel_threshold=args.pixel_threshold,
                lowest_healpix_order=4,
            )
            .add_margin(margin_threshold=10.0, is_default=True)
        )

        with Client(n_workers=min(8, args.num_processes), threads_per_worker=1) as client:
            pipeline_with_client(import_args, client)

        print(f"  Done: {survey_name}")

    print("All done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build SDSS parent sample in HATS format')
    parser.add_argument('sdss_data_path', type=str, help='Path to the local copy of the SDSS data')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    parser.add_argument('--num_processes', type=int, default=10)
    parser.add_argument('--pixel_threshold', type=int, default=8192,
                        help='Max rows per HATS partition')
    args = parser.parse_args()

    main(args)
