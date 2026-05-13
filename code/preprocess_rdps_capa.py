#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script preprocesses RDPS and CaPA_coarse data that will be used to create ORRC files. Preprocessing includes regridding the data, filling missing variables and files.

"""
import logging
import multiprocessing as mp
from copy import deepcopy
from datetime import date, timedelta
from os import getlogin
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

from utils import get_logs, classify_ncfiles, preprocess_dataset, check_files, create_file

logger = logging.getLogger('caspar_preprocess')
logger.setLevel(logging.INFO)


def main(version, tgt_file, all_vars, weights_dir, coordinate_vars, infolder=None, outfolder=None, logging_dir=None, overwrite=False):
    logfile = logging_dir.joinpath('preprocess_rdps_capa.log')
    logging_dir.mkdir(parents=True, exist_ok=True)
    # remove handlers for each version
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfile, level=logging.INFO)
       
    # load logs of previous successful jobs
    log = get_logs(logfile=logfile) # successful jobs must include 'finished'
    logging.info(f"ORRC version set to: {version}. Preprocessing RDPS, CaPA_coarse and CaPA_24h onto the target grid: {tgt_file}.")

    ds_tgt = xr.open_dataset(tgt_file).isel(time=0).drop_vars("time") # only need lat-lon 

    sub_dirs = [subdir for subdir in infolder.iterdir() if subdir.is_dir()]

    # loop over the input folder subdirectories
    for sub_dir in sub_dirs:
        print(sub_dir)

        all_ncfiles = list(sub_dir.glob('*.nc'))

        # group files by (year, month) and save in a dictionary
        # will look like ex: ('2020', '01'): [Path('20200101_file.nc'), Path('20200102_file.nc')]
        files_by_yr_mo = {}
        for nc in all_ncfiles:
            yr = nc.name[0:4]
            mo = nc.name[4:6]
            files_by_yr_mo.setdefault((yr, mo), []).append(nc)

        # get the unique years and months and sort them  
        years = sorted(set(yr for (yr, mo) in files_by_yr_mo.keys()))
        print(years)

        # preprocess in bunches of year / month
        for year in years:
            for month in range(1,13):
                # print(year, month)
                
                # skip year month treatment if already finished in logs
                logentry = f"{sub_dir.as_posix()}:{year}_{str(month).zfill(2)}"
                if logentry in log and not overwrite:
                    print(f"Already processed, skipping {logentry}")
                    continue

                # retrive files for the year-month
                crnt_files = files_by_yr_mo.get((year, f"{month:02d}"), [])

                if not crnt_files:
                    continue

                # classify files once
                # create a dictionary specification with output dataset coordinates (excluding time)
                # and filelist for each ECCC_grid_definition present in the files
                grid_spec = classify_ncfiles(ncfiles=crnt_files)

                for prd in grid_spec.keys():
                    print(f"prd: {prd}")
                    for eccc_grd in grid_spec[prd].keys():
                        # Create job list
                        jobs = []
                        # loop over the files in each eccc grid definition category
                        for nc in grid_spec[prd][eccc_grd]['filelist']:
                            spec = {"product":prd, "coords":deepcopy(grid_spec[prd][eccc_grd]['coords']), "label":eccc_grd}
                            # remove time coordinate from spec, it will be replaced by the timestamp for each file
                            if 'time' in spec['coords']:
                                del spec['coords']['time']
                            # all files will be saved in the same folder regardless of the original eccc_grid_definition
                            outfoler_nam = outfolder.joinpath(f"{prd}_regridded", "NAM") 
                            outfoler_nam.mkdir(parents=True, exist_ok=True)
                            outfile = outfoler_nam.joinpath(nc.name)
                            if not outfile.exists() or overwrite:
                                jobs.append((nc, spec, all_vars[prd], ds_tgt, weights_dir, outfile))
                                #preprocess_dataset(nc=nc, spec=spec, outfile=outfile)
                            else:
                                print(f'Processed file already exists, skipping {nc.name}')
                        if jobs:
                            preprocess_dataset(*jobs[0]) # process the first file without multiprocessing 
                            if len(jobs) > 1:
                                pool= mp.Pool(16)
                                pool.starmap(preprocess_dataset, jobs[1:])
                                pool.close()
                                pool.join()
                
                # gather all final netCDF files for this year-month 
                all_ncfiles_for_month = []
                for prd in grid_spec.keys():
                    outfoler_nam = outfolder.joinpath(f"{prd}_regridded", "NAM")
                    monthly_files = list(outfoler_nam.glob(f"{year}{month:02d}*.nc"))
                    all_ncfiles_for_month.extend(monthly_files)

                # if no regridded files found for that month, log incomplete
                if not all_ncfiles_for_month:
                    logger.info(f'incomplete:{logentry}')
                    print(f"Incomplete (no final files in outfolder): {logentry}")
                    continue

                # dictionary: YYYYMMDD -> set(hours)
                files_by_dayhour = {}
                for nc in all_ncfiles_for_month:
                    date_str = nc.name[:8]   # e.g. '20150101'
                    hour_str = nc.name[8:10] # e.g. '00','06','12','18'
                    files_by_dayhour.setdefault(date_str, set()).add(hour_str)

                # determine # of days in this month
                first_of_month = pd.Timestamp(year=int(year), month=int(month), day=1)
                days_in_month = (first_of_month + pd.offsets.MonthEnd(1)).day

                all_days_complete = True
                # RDPS and CaPA_coarse should have 4 files per day (00, 06, 12, 18), while CaPA_24h products only have 1 file per day (12)
                required_hours = {"00", "06", "12", "18"} if sub_dir.name in ['RDPS', 'CaPA_coarse'] else {"12"}
                # check for each day in the month if all 4 hours exist
                # when it finds any day that is missing altogether or missing some required hours, break and log incomplete
                for day_int in range(1, days_in_month + 1):
                    day_str = f"{day_int:02d}"
                    date_key = f"{year}{month:02d}{day_str}"  # e.g., 20150101
                    if date_key not in files_by_dayhour:
                        # entire day missing
                        all_days_complete = False
                        break
                    else:
                        hours_we_have = files_by_dayhour[date_key]
                        if not required_hours.issubset(hours_we_have):
                            all_days_complete = False
                            break

                if all_days_complete:
                    logger.info(f'finished:{logentry}')
                    print(f"Finished (all files found in outfolder): {logentry}")
                else:
                    logger.info(f'incomplete:{logentry}')
                    print(f"Incomplete (some days/hours missing in outfolder): {logentry}")
                # after missing files are filled, the next time the sript is run, it will find all files and log finished


    # go through the regridded RDPS and CaPA_coarse files, find missing files and fill them 
    RDPS_regridded_dir = outfolder.joinpath(f"RDPS_regridded", "NAM") 
    RDPS_regridded_dir.mkdir(parents=True, exist_ok=True)
    CaPA6h_regridded_dir = outfolder.joinpath(f"CaPA_coarse_regridded", "NAM") 
    CaPA6h_regridded_dir.mkdir(parents=True, exist_ok=True)
    CaPA24h_regridded_dir = outfolder.joinpath(f"CaPA_24h_regridded", "NAM")
    CaPA24h_regridded_dir.mkdir(parents=True, exist_ok=True)

    RDPS_files = sorted(list(RDPS_regridded_dir.glob('*.nc')))
    CaPA6h_files = sorted(list(CaPA6h_regridded_dir.glob('*.nc')))
    CaPA24h_files = sorted(list(CaPA24h_regridded_dir.glob('*.nc')))

    if RDPS_files and CaPA6h_files and CaPA24h_files:
        first_date_RDPS = RDPS_files[0].stem[0:8]
        last_date_RDPS = RDPS_files[-1].stem[0:8]

        first_date_CaPA6h = CaPA6h_files[0].stem[0:8]
        last_date_CaPA6h = CaPA6h_files[-1].stem[0:8]

        first_date_CaPA24h = CaPA24h_files[0].stem[0:8]
        last_date_CaPA24h = CaPA24h_files[-1].stem[0:8]

        # find the common date range between RDPS, CaPA_coarse and CaPA_24h
        first_date = max(int(first_date_RDPS), int(first_date_CaPA6h), int(first_date_CaPA24h))
        last_date = min(int(last_date_RDPS), int(last_date_CaPA6h), int(last_date_CaPA24h))

        start_date = date(
            int(str(first_date)[0:4]),
            int(str(first_date)[4:6]),
            int(str(first_date)[6:8])
        )
        end_date = date(
            int(str(last_date)[0:4]),
            int(str(last_date)[4:6]),
            int(str(last_date)[6:8])
        )

        # end date minus 1 day, otherwise it creates files from scratch for end_date+1 for CaPA_coarse (see check_files function)
        end_date = end_date - timedelta(days = 1)
        last_date = end_date.strftime('%Y%m%d') 

        max_processed_date = "19000101"  # default to an old date
        for entry in log:
            if entry.startswith("check_missing_files:"):
                processed_period = entry.split("check_missing_files:")[-1].strip()
                _, last_processed_date = processed_period.split('_')
                if last_processed_date > max_processed_date:
                    max_processed_date = last_processed_date # str

        if int(max_processed_date) >= int(last_date) and not overwrite:
            print(f"All files up to {last_date} are already processed. Skipping.")
            return

        if int(max_processed_date) >= int(first_date):
            # start from the day after the latest processed date
            new_start_date = date(int(max_processed_date[0:4]), int(max_processed_date[4:6]), int(max_processed_date[6:8])) + timedelta(days = 1)
        else:
            # no previous processing, start from the common first_date
            new_start_date = start_date

        if new_start_date > end_date:
            print("No new files to process since the last run.")
            return

        print(f"Checking for missing files in {new_start_date.strftime('%Y%m%d')} - {last_date}...")
        run_start = new_start_date

        all_valid_RDPS_files = []
        all_valid_CaPA6h_files = []
        all_valid_CaPA24h_files = []
        all_missing_RDPS_files = []
        all_missing_CaPA6h_files = []
        all_missing_CaPA24h_files = []

        while new_start_date <= end_date:
            
            # read the input netCDF files
            valid_RSPS_files, valid_CaPA6h_files, valid_CaPA24h_files, missing_RDPS_files, missing_CaPA6h_files, missing_CaPA24h_files = check_files(RDPS_regridded_dir, CaPA6h_regridded_dir, CaPA24h_regridded_dir, new_start_date)        
            
            all_valid_RDPS_files.extend(valid_RSPS_files)
            all_valid_CaPA6h_files.extend(valid_CaPA6h_files)
            all_valid_CaPA24h_files.extend(valid_CaPA24h_files)
            all_missing_RDPS_files.extend(missing_RDPS_files)
            all_missing_CaPA6h_files.extend(missing_CaPA6h_files)
            all_missing_CaPA24h_files.extend(missing_CaPA24h_files)

            # increment the start date
            new_start_date += timedelta(days = 1)
        
        # create the missing files
        create_file(all_missing_RDPS_files, all_valid_RDPS_files, coordinate_vars)
        create_file(all_missing_CaPA6h_files, all_valid_CaPA6h_files, coordinate_vars)
        create_file(all_missing_CaPA24h_files, all_valid_CaPA24h_files, coordinate_vars)

        # log entry for the whole period
        logentry_mfiles = f"check_missing_files:{run_start.strftime('%Y%m%d')}_{last_date}"
        logger.info(f'finished:{logentry_mfiles}')


if __name__ == '__main__':
    
    user = getlogin()
    config_path = Path(__file__).parent.joinpath('data', 'config_orrc.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)   

    coordinate_vars = config['Coordinates']['coordinate_vars']
    all_vars = config['Variables'] 

    infolder = Path(config['Files']['raw_dir'])
    outfolder = Path(config['Files']['processed_dir'].format(user=user))
    weights_dir = outfolder.joinpath('weights')

    overwrite = config['Settings']['preprocess_overwrite']    

    for version in config['Settings']['versions']:
        tgt_file = Path(config['Files'][f"tgt_file_{version.replace('.', '')}"])
        outfolder_version = outfolder.joinpath(f"{version.replace('.', '')}") 
        logging_dir = Path(config['Files']['logging_dir'].format(user=user)).joinpath(f"{version.replace('.', '')}")

        main(version, tgt_file, all_vars, weights_dir, coordinate_vars, infolder=infolder, outfolder=outfolder_version, logging_dir=logging_dir, overwrite=overwrite)
