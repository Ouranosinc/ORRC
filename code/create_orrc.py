#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script creates hourly ORRC data using RDPS and CaPA data. 

"""

from datetime import date, timedelta
from pathlib import Path   
import pandas as pd
import multiprocessing as mp
import yaml 
import os
import logging

from utils import get_logs, create_orrc

logger = logging.getLogger('create_orrc')
logger.setLevel(logging.INFO)


def main(version, processed_dir, required_time_window, var_list, tgt_file, logging_dir, analysis_bool=True, overwrite=False):
    logfile = logging_dir.joinpath('create_orrc.log')
    
    # remove handlers for each version
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfile, level=logging.INFO)     
  
    # load logs of previous successful jobs
    log = get_logs(logfile=logfile)

    logging.info(f"Creating ORRC version: {version}")
    logging.info(f"The overwrite option is set to {overwrite}")

    if analysis_bool: 
        logging.info("The precipitation analysis will be included in the output files ")
    else:
        logging.info("WARNING: The precipitation analysis will NOT be included in the output files ")
    
            
    RDPS_regridded_dir = processed_dir.joinpath(f"RDPS_regridded", "NAM") 
    RDPS_regridded_dir.mkdir(parents=True, exist_ok=True)
    CaPA_regridded_dir = processed_dir.joinpath(f"CaPA_coarse_regridded", "NAM")
    CaPA_regridded_dir.mkdir(parents=True, exist_ok=True)
    
    RDPS_files = sorted(list(RDPS_regridded_dir.glob('*.nc'))) 
    CaPA_files = sorted(list(CaPA_regridded_dir.glob('*.nc')))

    # create output netCDF file for ORRC
    output_dir = processed_dir.joinpath('CaSR_mimic/NAM') 
    output_dir.mkdir(parents=True, exist_ok=True)

    if RDPS_files and CaPA_files:
        first_date_RDPS = RDPS_files[0].stem[0:8]
        last_date_RDPS = RDPS_files[-1].stem[0:8]

        first_date_CaPA = CaPA_files[0].stem[0:8]
        last_date_CaPA = CaPA_files[-1].stem[0:8]

        # find the common date range between RDPS and CaPA
        first_date = max(int(first_date_RDPS), int(first_date_CaPA))
        last_date = min(int(last_date_RDPS), int(last_date_CaPA))
        
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

        # end date minus 1 day because RDPS files are at 06, 12, 18, 00 of the same day
        # and CaPA files are at 18 of the same day and at 00, 06, 12 of the next day
        end_date = end_date - timedelta(days = 1)
        last_date = end_date.strftime('%Y%m%d') 

        current_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        while current_ts <= end_ts:
            first_of_month = current_ts.replace(day=1)
            last_of_month = first_of_month + pd.offsets.MonthEnd(1)
            if last_of_month > end_ts:
                last_of_month = end_ts
            
            if first_of_month > end_ts:
                break
                
            month_str = first_of_month.strftime('%Y%m')
            month_logentry = f"month_{month_str}"

            if month_logentry in log and not overwrite:
                print(f'{month_str} has already been processed, skipping...')
                current_ts = last_of_month + timedelta(days = 1)
                continue

            jobs = []
            day_iter = first_of_month

            while day_iter <= last_of_month:
                day_date = day_iter.to_pydatetime().date()  # e.g., 2015-01-01

                # output NetCDF file name (ORRC)
                output_file_name = "".join(("RDPS_CaPA_", day_date.strftime('%Y%m%d'), "12.nc"))
                
                # skip creating ORRC file if it already exists
                outfile_nc       = os.path.join(output_dir, output_file_name)
                if os.path.exists(outfile_nc) and not overwrite:
                    print(f"File {outfile_nc} already exists. Skipping it.")
                else:
                    jobs.append((
                        RDPS_regridded_dir, 
                        CaPA_regridded_dir, 
                        tgt_file, 
                        day_date, 
                        outfile_nc, 
                        var_list, 
                        required_time_window, 
                        analysis_bool
                    ))

                day_iter += timedelta(days=1) # e.g., datetime.date(2015, 1, 2)

            if not jobs:
                print(f"No new jobs to process for {month_str}, continuing to next month...")
                current_ts = last_of_month + timedelta(days = 1)
                continue

            pool = mp.Pool(16)
            pool.starmap(create_orrc, jobs)
            pool.close()
            pool.join()

            # gather all final .nc files for this month
            monthly_files = sorted(output_dir.glob(f"RDPS_CaPA_{month_str}??12.nc"))

            if not monthly_files:
                logger.info(f'incomplete:{month_logentry}')
                print(f"Incomplete (no files): {month_logentry}")
            else:
                # Dictionary: YYYYMMDD; one file per day at 12 UTC
                files_by_day = {}
                for nc in monthly_files:
                    date_str = nc.name[10:18]   # e.g. '20150101' from 'RDPS_CaPA_2015010112.nc'
                    files_by_day.setdefault(date_str, set())

                # determine # of days in this month
                days_in_month = (first_of_month + pd.offsets.MonthEnd(1)).day
                all_days_complete = True 

                for day_int in range(1, days_in_month + 1):
                    day_str = f"{day_int:02d}"
                    date_key = f"{first_of_month.year}{first_of_month.month:02d}{day_str}"
                    if date_key not in files_by_day:
                        all_days_complete = False
                        break

                if all_days_complete:
                    logger.info(f'finished:{month_logentry}')
                    print(f"Finished (all daily files found): {month_logentry}")
                else:
                    logger.info(f'incomplete:{month_logentry}')
                    print(f"Incomplete (some days missing): {month_logentry}")
            current_ts = last_of_month + timedelta(days = 1)
        
    print("All mimic RDRS files have been created.")
    

if __name__ == '__main__':
    user = os.getlogin()

    config_path = Path(__file__).parent.joinpath('data', 'config_orrc.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)   

    var_list = [var_info['name'] for var_info in config['Variables']['RDPS']]
    required_time_window = config['Time']['required_rdps_time_window']
    required_time_window = [int(i) for i in required_time_window] 

    analysis_bool = config['Settings']['precipitation_analysis_bool']
    overwrite = config['Settings']['create_orrc_overwrite']

    for version in config['Settings']['versions']:
        tgt_file = Path(config['Files'][f'tgt_file_{version.replace(".", "")}'])
        processed_dir = Path(config['Files']['processed_dir'].format(user=user)).joinpath(f"{version.replace('.', '')}") 
        logging_dir = Path(config['Files']['logging_dir'].format(user=user)).joinpath(f"{version.replace('.', '')}")

        main(version, processed_dir, required_time_window, var_list, tgt_file, logging_dir=logging_dir, analysis_bool=analysis_bool, overwrite=overwrite)