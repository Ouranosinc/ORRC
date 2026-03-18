#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script converts the ORRC NetCDF files to CF-compliant NetCDF files using the Miranda library and saves them as zarr.zip files, both in hourly and daily format.

"""

from pathlib import Path
import yaml 
import os
import xarray as xr
import xscen as xs
import logging
import sys

from miranda.convert.eccc_rdrs import convert_rdrs, rdrs_to_daily
from utils import split_ds, zarr_to_zarr_zip, get_datelen_format, remove_partial_zarrs, mask_data

logger = logging.getLogger('convert_orrc')
logger.setLevel(logging.INFO)


def main(version, base_dir, staging_dir, working_dir, logging_dir, variables, daily_agg=None, mask=None, overwrite=False):
    logfile = logging_dir.joinpath('convert_orrc.log')
    
    # remove handlers for each version
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )   

    logging.info(f"Converting ORRC version: {version}")


    home = Path("~").expanduser()
    dask_dir = home.joinpath("my_dir", "tmp_orrc", "dask")
    dask_dir.mkdir(parents=True, exist_ok=True)
    dask_kwargs = dict(
        n_workers=6,
        threads_per_worker=6, #6
        memory_limit="12GB",
        dashboard_address=8991,
        local_directory=dask_dir,
        silence_logs=logging.ERROR,
    )
    project = f"ORRC-{version.replace('.', '')}"      

    outfolder = base_dir.joinpath(f"CaSR_mimic", "tmp", "NAM")
    year_start_hr  = None
    year_start_day = None
    for var in variables:
        hr_var_dir = outfolder.joinpath("1hr", var)
        hr_var_dir.mkdir(parents=True, exist_ok=True)
        zarr_files = sorted(hr_var_dir.glob("*.zarr"))
        if zarr_files:
            last_zarr = zarr_files[-1]
            last_year_hr = last_zarr.stem.split("_")[-1][:4]
            year_start_hr = int(last_year_hr) if year_start_hr is None else min(year_start_hr, int(last_year_hr))
        else:
            year_start_hr = None
        
        day_var_dir = outfolder.joinpath("day", var)
        day_var_dir.mkdir(parents=True, exist_ok=True)
        zarr_files = sorted(day_var_dir.glob("*.zarr"))
        if zarr_files:
            last_zarr = zarr_files[-1]
            last_year_day = last_zarr.stem.split("_")[-1][:4]
            year_start_day = int(last_year_day) if year_start_day is None else min(year_start_day, int(last_year_day))
        else:
            year_start_day = None
    logger.info(f"Year start for converting hourly files: {year_start_hr}")
    logger.info(f"Year start for converting daily files: {year_start_day}")
    

    # save the converted files in Zarr format to "1hr" sub-dir
    convert_rdrs(
        project=project,
        input_folder=base_dir.joinpath(f"CaSR_mimic", "NAM"),
        output_folder=base_dir.joinpath(f"CaSR_mimic", "tmp", "NAM"),
        output_format="zarr",
        working_folder=home.joinpath("my_dir", "tmp_orrc", "orrc"),
        overwrite=overwrite, 
        year_start=year_start_hr, 
        year_end=None,
        cfvariable_list=variables,
        **dask_kwargs,
    )
    logger.info("Converted all variables to CF-compliant format and saved them as Zarr files.")

    # convert from 1hr to daily
    rdrs_to_daily(
        project=project,
        input_folder=base_dir.joinpath(f"CaSR_mimic", "tmp", "NAM", "1hr"),
        output_folder=base_dir.joinpath(f"CaSR_mimic", "tmp", "NAM", "day"),
        working_folder=home.joinpath("my_dir", "tmp_orrc", "orrc1"),
        overwrite=overwrite,
        year_start=year_start_day, 
        year_end=None,
        process_variables=variables,
        **dask_kwargs,
    )
    logger.info("Converted all variables to daily format and saved them as Zarr files.")
    
    for freq in ["1hr", "day"]: 
        infolder = base_dir.joinpath(f"CaSR_mimic", "tmp", "NAM", freq) 
        infolder.mkdir(parents=True, exist_ok=True)
        freq_variables = list(variables)
        if freq == "day" and daily_agg is not None:
            freq_variables = freq_variables + list(daily_agg)

        freq_mask = None
        if mask is not None:
            freq_mask = {
                "variables": mask.get("variables", []),
                "start_time": mask.get(freq, {}).get("start_time"),
                "end_time": mask.get(freq, {}).get("end_time"),
            }

        for var_dir in infolder.iterdir():
            if var_dir.is_dir() and var_dir.name in freq_variables:
                variable = var_dir.name   
                paths = sorted(var_dir.glob("*.zarr"))
                unzipped_flag = any([not(str(p).endswith(".zarr.zip")) for p in paths])
                fmt = 'zarr'  # assuming all files are in zarr format
                if unzipped_flag and fmt == 'zarr':
                    if len(paths) > 1:  # concatenate files
                        dsdct = {variable: xr.open_mfdataset(paths, engine="zarr", decode_timedelta=False)}
                        assert len(dsdct.keys()) == 1
                        for key, ds in dsdct.items():
                            for attr in [d for d in ds.attrs if d in ['date_start', 'date_end']]:
                                del ds.attrs[attr]
                            for attr in [d for d in ds.attrs if d == 'processing_level']:
                                ds.attrs[attr] = 'raw'              # for the end user, the processing level is 'raw' 
                            assert len(ds.data_vars) == 1
                            assert variable in ds.data_vars
                            if freq == 'day':
                                xrfreq = "D"
                                ndays = 360 if ds.time.dt.calendar == '360_day' else 365
                                timechunk = ndays * 4
                                if ds.time.dt.calendar not in ['noleap', '365_day']:
                                    timechunk += 1
                            elif freq == '1hr':  
                                xrfreq = "h"
                                timechunk = 1440
                            facets = {
                                'version': f"{ds.attrs['version'].replace('.','')}",
                                } 

                            for dsout in split_ds(ds):
                                if dsout is not None:
                                    date_start = dsout.time.min().dt.strftime('%Y%m%d').item()
                                    outpath = xs.catutils.build_path(dsout, root=staging_dir, format=fmt, **facets)

                                    # fill the data with NaN based on config variables and time period
                                    dsout = mask_data(dsout, variable, freq_mask)

                                    if not outpath.with_suffix('.zarr.zip').exists():
                                        outpath.parent.mkdir(parents=True, exist_ok=True)
                                        zarr_to_zarr_zip(dsout, outpath, variable, timechunk, fmt, working_dir, dask_kwargs)
                                        logging.info(f"Successfully created new file {outpath.with_suffix('.zarr.zip')}.")
                                    else:
                                        # check if there is an existing file with the same start year but fewer time steps
                                        datelen, date_fmt = get_datelen_format(outpath)
                                        existing_zarrzip = list(outpath.parent.glob(f"{variable}_*_{dsout.time.min().dt.strftime(date_fmt).item()}-*.zarr.zip"))
                                        if existing_zarrzip:
                                            assert len(existing_zarrzip) == 1, f"Multiple zarr.zip files with the same start date in {outpath.parent}. Need to investigate."
                                            existing_ds = xr.open_zarr(existing_zarrzip[0])
                                            # verify that the existing file has the same start date
                                            if "time" in dsout.dims and "time" in existing_ds.dims and date_start == existing_ds.time.min().dt.strftime('%Y%m%d').item():
                                                if len(dsout.time) > len(existing_ds.time):
                                                    try:
                                                        os.remove(existing_zarrzip[0])
                                                        zarr_to_zarr_zip(dsout, outpath, variable, timechunk, fmt, working_dir, dask_kwargs)
                                                        logger.info(f"{outpath.with_suffix('.zarr.zip')} is longer than existing {existing_zarrzip[0]}; overwriting.")
                                                    except Exception as e:
                                                        logger.error(f"Error removing existing file {existing_zarrzip[0]}: {e}")
                                                        sys.exit("Unable to remove existing file. Exiting.")
                                                elif len(dsout.time) == len(existing_ds.time): # same check as the filename check, but added for safety
                                                    if overwrite:
                                                        try:
                                                            os.remove(existing_zarrzip[0])
                                                            zarr_to_zarr_zip(dsout, outpath, variable, timechunk, fmt, working_dir, dask_kwargs)
                                                            logger.info(f"{outpath.with_suffix('.zarr.zip')} has the same number of time points as existing {existing_zarrzip[0]}; and overwrite=True; overwriting.")
                                                        except Exception as e:
                                                            logger.error(f"Error removing existing file {existing_zarrzip[0]}: {e}")
                                                            sys.exit("Unable to remove existing file. Exiting.")
                                                    else:
                                                        logger.info(f"Skipping {outpath.with_suffix('.zarr.zip')} as it has the same number of time points as existing file and and overwrite=False.")
                                                        continue
                                                else: # this scenario should not happen, but added for safety
                                                    logger.warning(f"Skipping {outpath.with_suffix('.zarr.zip')} as it has fewer time points than existing file. Need to investigate.")
                                                    continue
                                            existing_ds.close()
                                            
                                    # remove any partial zarrs covered by complete years; should only occur for first run of the year
                                    remove_list = remove_partial_zarrs(outpath.parent)
                                    if remove_list:
                                        for p in remove_list:
                                            p.unlink()
                                            logging.info(f"Removed partial zarr file covered by complete year: {p}")

                    # if only one zarr convert directly to zarr.zip
                    else:
                        ds = xr.open_zarr(paths[0], decode_timedelta=False)

                        for attr in [d for d in ds.attrs if d in ['date_start', 'date_end']]:
                            del ds.attrs[attr]
                        for attr in [d for d in ds.attrs if d == 'processing_level']:
                            ds.attrs[attr] = 'raw'

                        if freq == 'day':
                            ndays = 360 if ds.time.dt.calendar == '360_day' else 365
                            timechunk = ndays * 4
                            if ds.time.dt.calendar not in ['noleap', '365_day']:
                                timechunk += 1
                        elif freq == '1hr':
                            timechunk = 1440

                        facets = {
                            'version': f"{ds.attrs['version'].replace('.','')}",
                        }

                        ds = mask_data(ds, variable, freq_mask)

                        outpath_sgl = xs.catutils.build_path(ds, root=staging_dir, format=fmt, **facets)

                        if not outpath_sgl.with_suffix(".zarr.zip").exists() or overwrite:
                            outpath_sgl.parent.mkdir(parents=True, exist_ok=True)
                            zarr_to_zarr_zip(ds, outpath_sgl, variable, timechunk, fmt, working_dir, dask_kwargs)
                            logger.info(f"Successfully created {outpath_sgl.with_suffix('.zarr.zip')}.")
                        else:
                            logger.info(f"File {outpath_sgl.with_suffix('.zarr.zip')} already exists and overwrite=False. Skipping.")


if __name__ == "__main__":
    user = os.getlogin()

    config_path = Path(__file__).parent.joinpath('data', 'config_orrc.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)   
        
    staging_dir = Path(config['Files']['staging_dir'].format(user=user))
    working_dir = Path(config['Files']['working_dir'].format(user=user))
    working_dir.mkdir(parents=True, exist_ok=True)

    variables = config['Target_variables']['Hourly']
    daily_agg = config['Target_variables']['Daily_aggregates']

    overwrite = config['Settings']['convert_orrc_overwrite']

    mask = config['Settings']['mask']

    for version in config['Settings']['versions']:
        base_dir = Path(config['Files']['processed_dir'].format(user=user)).joinpath(f"{version.replace('.', '')}") 
        logging_dir = Path(config['Files']['logging_dir'].format(user=user)).joinpath(f"{version.replace('.', '')}")

        main(version, base_dir, staging_dir, working_dir, logging_dir, variables, daily_agg=daily_agg, mask=mask, overwrite=overwrite)