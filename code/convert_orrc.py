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
import shutil

from datetime import datetime
from tempfile import TemporaryDirectory

from miranda.convert.eccc_rdrs import convert_rdrs, rdrs_to_daily
from xclim import convert
from utils import split_ds, zarr_to_zarr_zip, get_datelen_format, remove_partial_zarrs, mask_data

logger = logging.getLogger('convert_orrc')
logger.setLevel(logging.INFO)


def main(version, base_dir, staging_dir, working_dir, logging_dir, variables, exclude_daily=None, add_daily=None, mask=None, overwrite=False):
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
    hrly_vars_to_aggregate = [v for v in variables if v not in exclude_daily]

    # convert from 1hr to daily
    rdrs_to_daily(
        project=project,
        input_folder=base_dir.joinpath(f"CaSR_mimic", "tmp", "NAM", "1hr"),
        output_folder=base_dir.joinpath(f"CaSR_mimic", "tmp", "NAM", "day"),
        working_folder=home.joinpath("my_dir", "tmp_orrc", "orrc1"),
        overwrite=overwrite,
        year_start=year_start_day, 
        year_end=None,
        process_variables=hrly_vars_to_aggregate,
        **dask_kwargs,
    )
    logger.info("Converted all variables to daily format and saved them as Zarr files.")
    
    fmt = 'zarr'  # assuming all files are in zarr format
    for freq in ["1hr", "day"]: 
        infolder = base_dir.joinpath(f"CaSR_mimic", "tmp", "NAM", freq) 
        infolder.mkdir(parents=True, exist_ok=True)
        freq_variables = list(variables)
        if freq == "day" and add_daily is not None:
            freq_variables = freq_variables + list(add_daily)

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

                                    date_start = dsout.time.min().dt.strftime("%Y%m%d").item()
                                    date_end = dsout.time.max().dt.strftime("%Y%m%d").item()

                                    facets = dict(base_facets)
                                    facets["date_start"] = date_start
                                    facets["date_end"] = date_end

                                    # fill the data with NaN based on config variables and time period
                                    dsout = mask_data(dsout, variable, freq_mask)
                                    dsout.attrs["xrfreq"] = xrfreq

                                    outpath = xs.catutils.build_path(dsout, root=staging_dir, format=fmt, **facets)
                                    outzip = outpath.with_suffix('.zarr.zip')
                                    if not outzip.exists() or overwrite:
                                        outpath.parent.mkdir(parents=True, exist_ok=True)

                                        if outzip.exists() and overwrite:
                                            outzip.unlink()

                                        zarr_to_zarr_zip(dsout, outpath, variable, timechunk, fmt, working_dir, dask_kwargs)
                                        logging.info(f"Successfully created new file {outzip}.")
  
                                    # remove any partial zarrs already covered 
                                    remove_list = remove_partial_zarrs(outpath.parent)
                                    if remove_list:
                                        for p in remove_list:
                                            p.unlink()
                                            logging.info(f"Removed partial zarr file already covered: {p}")

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
                        
                        xrfreq = "h" if freq == "1hr" else "D"

                        facets = {
                            "version": f"{ds.attrs['version'].replace('.', '')}",
                            "xrfreq": xrfreq,
                            "date_start": ds.time.min().dt.strftime("%Y%m%d").item(),
                            "date_end": ds.time.max().dt.strftime("%Y%m%d").item(),
                        }

                        ds.attrs["xrfreq"] = xrfreq
                        ds = mask_data(ds, variable, freq_mask)
                        outpath_sgl = xs.catutils.build_path(ds, root=staging_dir, format=fmt, **facets)
                        outzip_sgl = outpath_sgl.with_suffix('.zarr.zip')

                        if not outzip_sgl.exists() or overwrite:
                            outpath_sgl.parent.mkdir(parents=True, exist_ok=True)

                            if outzip_sgl.exists() and overwrite:
                                outzip_sgl.unlink()

                            outpath_sgl.parent.mkdir(parents=True, exist_ok=True)
                            zarr_to_zarr_zip(ds, outpath_sgl, variable, timechunk, fmt, working_dir, dask_kwargs)
                            logger.info(f"Successfully created {outzip_sgl}.")
                        
                        remove_list = remove_partial_zarrs(outpath_sgl.parent)
                        if remove_list:
                            for p in remove_list:
                                p.unlink()
                                logging.info(f"Removed partial zarr file already covered: {p}")
                    

    
    # compute daily wind direction from vector components
    indir = staging_dir.joinpath(f"reconstruction/NAM/Ouranos/ORRC_{version.replace('.', '')}/day")

    wind_dicts = [
        {'winddir': ['uas', 'vas']}
    ]

    for variables in wind_dicts:
        for outvar, vv in variables.items():

            u_files = sorted(indir.joinpath(vv[0]).glob("*.zarr.zip"))
            v_files = sorted(indir.joinpath(vv[1]).glob("*.zarr.zip"))

            if not u_files or not v_files:
                logging.warning(f"No zarr.zip files found for variables {vv} in {indir}. Skipping computation for {outvar}.")
                continue

            ds_u = xr.open_mfdataset(u_files, engine="zarr", decode_timedelta=False)
            ds_v = xr.open_mfdataset(v_files, engine="zarr", decode_timedelta=False)

            out = convert.wind_speed_from_vector(
                uas=ds_u[vv[0]],
                vas=ds_v[vv[1]],
            ).sfcWindfromdir


            glob_attrs = ds_u.attrs.copy()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            glob_attrs["history"] = (f"{timestamp}: Computed daily wind direction from {vv[0]} and {vv[1]} using xclim.convert.wind_speed_from_vector.")

            dsout = xr.Dataset(attrs=glob_attrs)
            dsout[outvar] = out

            ndays = 360 if dsout.time.dt.calendar == "360_day" else 365
            timechunk = ndays * 4
            if dsout.time.dt.calendar not in ["noleap", "365_day"]:
                timechunk += 1
            chunks = {d:50 for d in dsout[outvar].dims if d != 'time'} 
            chunks["time"] = timechunk

            dsout = xs.io.rechunk_for_saving(dsout, rechunk=chunks)

            date_start = dsout.time.min().dt.strftime('%Y%m%d').item()
            date_end = dsout.time.max().dt.strftime('%Y%m%d').item()
            facets = {
                "domain": "NAM",
                "version": version.replace('.', ''),
                "date_start": date_start,
                "date_end": date_end,
                "xrfreq": "D",
            }

            outpath = xs.catutils.build_path(dsout, root=staging_dir, schemas=None, format=fmt, **facets)
            outpath = Path(str(outpath).replace(" ", "_"))
            outzip = outpath.with_suffix(".zarr.zip")
            if outzip.exists():
                logging.info(f"{outzip} already exists. Skipping.")
                continue

            with TemporaryDirectory(dir=working_dir) as tmpdir:
                tmpzarr = Path(tmpdir).joinpath(outpath.name)
                tmpzip = tmpzarr.with_suffix(".zarr.zip")

                xs.io.save_to_zarr(dsout, tmpzarr, mode="w")
                xs.io.zip_directory(root=tmpzarr, zipfile=tmpzip, delete=True)
                outzip.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(tmpzip, outzip)

            del dsout
            logging.info(f"Computed daily wind direction for {outvar} and saved to {outzip}.")


if __name__ == "__main__":
    user = os.getlogin()

    config_path = Path(__file__).parent.joinpath('data', 'config_orrc.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)   
        
    staging_dir = Path(config['Files']['staging_dir'].format(user=user))
    working_dir = Path(config['Files']['working_dir'].format(user=user))
    working_dir.mkdir(parents=True, exist_ok=True)

    variables = config['Target_variables']['Hourly']
    exclude_daily = config['Target_variables']['Exclude_daily']
    add_daily = config['Target_variables']['Add_daily']

    overwrite = config['Settings']['convert_orrc_overwrite']

    mask = config['Settings']['mask']

    for version in config['Settings']['versions']:
        base_dir = Path(config['Files']['processed_dir'].format(user=user)).joinpath(f"{version.replace('.', '')}") 
        logging_dir = Path(config['Files']['logging_dir'].format(user=user)).joinpath(f"{version.replace('.', '')}")

        main(version, base_dir, staging_dir, working_dir, logging_dir, variables, exclude_daily=exclude_daily, add_daily=add_daily, mask=mask, overwrite=overwrite)