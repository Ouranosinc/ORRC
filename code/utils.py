#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script contains utility functions used in the ORRC workflow.

"""

import logging
import os
import shutil
import sys
from datetime import datetime, timedelta
from os.path import join
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, Union

import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client

import xscen as xs

logger = logging.getLogger('caspar_preprocess')
logger.setLevel(logging.INFO)


def get_logs(logfile : str) -> list:
    """
    Extracts the log entries from the log file.

    Parameters
    ----------
    logfile : str
        The path to the log file.
    
    Returns
    -------
    log : list
        The list of log entries.
    """
    log = []
    with open(logfile, 'r') as f:
        for line in f:
            if "INFO:caspar_preprocess:finished:" in line:
                line = line.replace("INFO:caspar_preprocess:finished:", "").strip()
                log.append(line)
            elif "INFO:create_orrc:finished:" in line:
                line = line.replace("INFO:create_orrc:finished:", "").strip()
                log.append(line)
    return log


def check_variables(ds: xr.Dataset, var_list: list, remove_vars: list = None) -> Tuple[xr.Dataset, bool]:
    """
    Checks if the variables in the var_list are present in the dataset. If not, it adds them as empty arrays with the provided attributes.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to check.
    var_list : list
        The list of variables to check.
    remove_vars : list, optional
        The list of variables to remove from the dataset. The default is None.
    
    Returns
    -------
    ds : xr.Dataset
        The dataset with the variables added.
    missing_boolean : bool
        A boolean indicating if any of the variables were missing.
    """
    missing_boolean = False
    for var_info in var_list:
        var_name = var_info['name']
        attrs = var_info.get('attributes', {})
        if var_name not in ds.data_vars:
            ds[var_name] = xr.DataArray(coords=ds.coords) 
            ds[var_name].attrs.update(attrs) # add attributes
            logging.info(f"Variable '{var_name}' not found in the dataset. Added as an empty array with attributes")
            missing_boolean = True
    if remove_vars is not None:
        for var_name in remove_vars:
            if var_name in ds.data_vars:
                ds = ds.drop_vars(var_name)
    return ds, missing_boolean


def classify_ncfiles(ncfiles : list) -> dict:
    """
    Classifies the netCDF files based on the product and eccc grid definition.

    Parameters
    ----------
    ncfiles : list
        List of the netCDF files to classify.
    
    Returns
    -------
    grid_spec : dict
        The dictionary containing the classification of the netCDF files.
    """
    grid_spec = {}
    
    for nc_path in ncfiles:
        with xr.open_dataset(nc_path, decode_timedelta=False) as ds:
            product = ds.attrs['product']
            #eccc_grd = ds['lon'].attrs['eccc_grid_definition'].replace(' ','_').replace(':','').replace(',','')
            dims = [d for d in ds.dims if d != 'time']
            if not np.all([len(ds[d].shape)==1 for d in dims]):
                raise ValueError(f'found dataset dimensions which are not 1D in {nc_path}..')
            eccc_grd= '_'.join([f"{d}{ds[d].shape[0]}_res{str(ds[d].diff(dim=d).mean().round(3).values).zfill(4)}".replace('.','') for d in dims])
            if 'time' in ds.dims:
                coords = ds.drop_dims('time').coords
        
            if product not in grid_spec.keys():
                grid_spec[product] = {}
            if eccc_grd not in grid_spec[product].keys():
                grid_spec[product][eccc_grd] = {"coords": coords, "filelist":[]}
            else:
                # make sur coords are close
                for c, val in grid_spec[product][eccc_grd]['coords'].items():
                    if 'lon' in c or 'lat' in c:
                        np.testing.assert_array_almost_equal(val, ds[c], 3)  

            grid_spec[product][eccc_grd]['filelist'].append(nc_path)
    
    return grid_spec


def add_time(nc : Path, spec : dict) -> xr.Dataset:
    """
    Adds the time dimension to the dataset based on the filename.

    Parameters
    ----------
    nc : Path
        The path to the NetCDF file.
    spec : dict
        Dictionary with:
          - 'product': 'rdps' or 'capa_coarse'
          - 'coords': additional coords (without 'time')

    Returns
    -------
    dsout : xr.Dataset
        The dataset with the time dimension and variable added as integer hours.
    """
    # parse timestamp from filename
    ref_time = pd.to_datetime(nc.stem, format="%Y%m%d%H")

    ds = xr.open_dataset(nc, decode_timedelta=False)

    if spec['product'].lower() == 'capa_coarse':
        time_values = np.array([0], dtype='int32')        # just [0]
    elif spec['product'].lower() == 'rdps':
        time_values = np.arange(6, 13, 1, dtype='int32')  # [6..12]
    else:
        raise ValueError(f"Unknown product {spec['product']}")
    
    # print(f'File: {nc}')
    if "time" not in ds.dims:
        ds = ds.expand_dims(dim={"time": len(time_values)})  # add a new dimension with the appropriate size
        ds = ds.assign_coords(time=("time", time_values))
    else: 
        # convert datetimes to "hours since ref_time"
        # e.g., file name is 2021010100.nc, then 2021-01-01 06:00:00 -> 6
        # ds.time will now be integer-based
        hours_since = (ds["time"].values - np.datetime64(ref_time)) / np.timedelta64(1, "h")
        hours_since = hours_since.astype("int32") 
        try:
            ds = ds.assign_coords(time=("time", hours_since))
        except ValueError as ve:
            logging.error(f"Error assigning time values to {nc}: {ve}")
        # reindex will fill the time dimension with NaN for missing values
        ds = ds.reindex(time=time_values)

        # explicitly overwrite the time values with the integer hours
        ds['time'] = ('time', time_values)

        for var in ds.data_vars:
            if var == 'RDPS_P_PR_SFC':
                # fill the missing values with the previous value
                ds[var] = ds[var].ffill(dim='time')
    
    ds['time'].attrs['units'] = f"hours since {ref_time.strftime('%Y-%m-%d %H:%M:%S')}"
    ds['time'].attrs['calendar'] = "gregorian"
    ds['time'].attrs['long_name'] = "time"
    ds['time'].attrs['standard_name'] = "time"
    ds['time'].attrs['axis'] = "T"

    for cvar, cdata in spec['coords'].items():
        ds = ds.assign_coords({cvar: cdata})

    dsout = ds.copy()
    ds.close()
    return dsout


def regrid_and_save(infile : Path, ds_in : xr.Dataset, ds_tgt : xr.Dataset, weights_location : Path, outfile : Path) -> None:
    """
    Regrids the input dataset onto the target grid and saves the regridded file.

    Parameters
    ----------
    infile : Path
        The input file to regrid.
    ds_in : xr.Dataset
        The input dataset to regrid.
    ds_tgt : xr.Dataset
        The target dataset to regrid onto.
    weights_location : Path
        The directory to save the weights file.
    outfile : Path
        The output file to save the regridded dataset.
    
    Returns
    -------
    None
    """
    product = ds_in.attrs['product']
    #ds_in.attrs["cat:id"] = f"{product}_{ds_in['lat'].attrs['eccc_grid_definition']}"
    ds_tgt.attrs["cat:id"] = "CaSR_v31_gridA"  # target grid is consistent

    ds_in.attrs["cat:domain"] = "input_domain"
    ds_tgt.attrs["cat:domain"] = "target_domain"


    try:
        # if 'rlat' in ds_in_trimmed.coords and 'rlon' in ds_in_trimmed.coords:
        #     method = "conservative_normed"
        # else: 
        
        method = "nearest_s2d"
        
        weights_location.mkdir(parents=True, exist_ok=True)
        weights_file = weights_location.joinpath(f"weights_{ds_in.attrs['cat:id']}_{ds_tgt.attrs['cat:domain']}_regrid0{method}.nc")
        ds_regridded = xs.regrid.regrid_dataset(ds_in, ds_tgt, 
                                                regridder_kwargs = {"method": method, "skipna": True, "reuse_weights": weights_file.exists()}, 
                                                weights_location=str(weights_location))

        # add history attribute
        ds_regridded.attrs['history'] = f"Regridded onto the CaSR v3.1 grid using xscen.regrid.regrid_dataset with the method {method}."

    except Exception as e:
        logging.error(f"Error regridding {infile}: {e}")
        return

    # remove 'cat:domain' attribute
    for var_name in ds_regridded.data_vars:
        if 'cat:domain' in ds_regridded[var_name].attrs:
            del ds_regridded[var_name].attrs['cat:domain']
    if 'cat:domain' in ds_regridded.attrs:
        del ds_regridded.attrs['cat:domain']

    # compress all variables using zlib with a compression level of 2 (1 is fastest, 9 is slowest)
    comp = dict(zlib=True, complevel=2)
    encoding = {var: comp for var in ds_regridded.data_vars}

    with ProgressBar():
        outfile.parent.mkdir(exist_ok=True, parents=True)
        ds_regridded.to_netcdf(outfile.with_suffix('.tmp.nc'), encoding=encoding, engine='netcdf4') # h5netcdf
    shutil.move(outfile.with_suffix('.tmp.nc'), outfile)


def preprocess_dataset(nc : Path, spec : dict, var_list : list, ds_tgt : xr.Dataset, weights_location : Path, outfile : Path) -> None:
    """
    Preprocesses the input dataset, regrids and saves it to the output file.

    Parameters
    ----------
    nc : Path
        The path to the NetCDF file.
    spec : dict
        Dictionary with:
            - 'product': 'rdps' or 'capa_coarse'
            - 'coords': additional coords (without 'time')
    var_list : list
        The list of variables to check.
    ds_tgt : xr.Dataset
        The target dataset to regrid onto.
    weights_location : Path
        The directory to save the weights file.
    outfile : Path
        The output file to save the regridded dataset.
    
    Returns
    -------
    None
    """
    dsout = add_time(nc, spec)
    remove_vars = [v for v in dsout.data_vars if v not in [v['name'] for v in var_list] and v != 'rotated_pole']
    dsout, missing_boolean = check_variables(dsout, var_list, remove_vars=remove_vars)
    if missing_boolean:
        logging.warning(f"Missing variables in {nc}. Added as empty arrays.")
    dsout.attrs["cat:id"] = f"{spec['product']}_{spec['label']}"
    regrid_and_save(nc, dsout, ds_tgt, weights_location, outfile)


def _add_history_attr(nc_file : nc4.Dataset, message : str) -> None:
    """
    Adds a history attribute to the given NetCDF file.
    Appends the provided message to the existing history.

    Parameters
    ----------
    nc_file : netCDF4.Dataset
        The NetCDF file object.
    message : str
        The message to append to the history attribute.

    Returns
    -------
    None.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_message = f"{current_time}: {message}"
    
    if 'history' in nc_file.ncattrs():
        nc_file.history += f"\n{history_message}"
    else:
        setattr(nc_file, 'history', history_message)


def check_files(RDPS_directory : str, CaPA_directory : str, start_date : datetime) -> Tuple[list, list, list, list]:
    """
    Checks if the required RDPS and CaPA files exist and have dimensions.

    Parameters
    ----------
    RDPS_directory : str
        path of the rdps files.
    CaPA_directory : str
        path of the capa files.
    start_date : str
        start date in the format of YYYYMMDD

    Returns
    -------
    valid_RDPS_files : list
        list of the valid RDPS file paths (files that exist and have dimensions).
    valid_CaPA_files : list
        list of the valid CaPA file paths (files that exist and have dimensions).
    missing_RDPS_files : list
        list of the missing RDPS file paths (either the file does not exist or it has no dimensions).
    missing_CaPA_files : list
        list of the missing CaPA file paths (either the file does not exist or it has no dimensions).

    """
 
    # turn the start_date datetime object into a string
    start_date_string = start_date.strftime("%Y%m%d")
    
    # also get the string for the date following the start date
    next_date         = start_date + timedelta(days = 1)
    next_date_string  = next_date.strftime("%Y%m%d")
    
    # create empty list to store the RDPS, CaPA and the name of files
    valid_RDPS_files  = []
    valid_CaPA_files  = []
    namefiledict    = {}
    missing_RDPS_files = []
    missing_CaPA_files = []
    
    # list of needed files from RDPS and CaPA
    namefiledict['RDPS'] = [os.path.join(RDPS_directory, start_date_string + "06.nc"),
                            os.path.join(RDPS_directory, start_date_string + "12.nc"),
                            os.path.join(RDPS_directory, start_date_string + "18.nc"),
                            os.path.join(RDPS_directory, next_date_string  + "00.nc")]
    
    namefiledict['CaPA'] = [os.path.join(CaPA_directory, start_date_string + "18.nc"),
                            os.path.join(CaPA_directory, next_date_string + "00.nc"),
                            os.path.join(CaPA_directory, next_date_string + "06.nc"),
                            os.path.join(CaPA_directory, next_date_string + "12.nc")]
            
    # Check if all the required RDPS files are present
    for file in namefiledict['RDPS']: 
        try:
            with nc4.Dataset(file, 'r') as dataset:
                if not dataset.dimensions:
                    missing_RDPS_files.append(file)
                    logging.error(f"RDPS file '{file}' exists but it has no dimensions.")
                else:
                    valid_RDPS_files.append(file)  # store valid RDPS file paths
                    # logging.info(f"RDPS file {file} is present --> OK")

        except FileNotFoundError:
            missing_RDPS_files.append(file)
            logging.error(f"RDPS file not found: {file}")

        except OSError as e:
            missing_RDPS_files.append(file)
            logging.error(f"Problem opening RDPS file: {file}, Error: {e}")
            
    
    # Check if all CaPA files are present
    for file in namefiledict['CaPA']: 
        try:
            with nc4.Dataset(file, 'r') as dataset:
                if not dataset.dimensions:
                    missing_CaPA_files.append(file)
                    logging.error(f"CaPA file '{file}' exists but it has no dimensions.")
                else:
                    valid_CaPA_files.append(file)  
                    # logging.info(f"CaPA file {file} is present --> OK")

        except FileNotFoundError:
            missing_CaPA_files.append(file)
            logging.error(f"CaPA file not found: {file}")
            
        except OSError as e:
            missing_CaPA_files.append(file)
            logging.error(f"Problem opening CaPA file: {file}, Error: {e}")

    return valid_RDPS_files, valid_CaPA_files, missing_RDPS_files, missing_CaPA_files


def create_file(missing_filepaths : list, complete_filepaths : list, coordinate_vars : list) -> None:
    """
    Creates new NetCDF file with the same variables from a complete file and fills the data with NaN.

    Parameters
    ----------
    missing_filepaths : list of Path
        Paths to the files that do not exist and need to be created.
    complete_filepaths : list of Path
        Paths to the files that contain all the required variables (used as templates).
    coordinate_vars : list of str
        Variables that are coordinates (e.g., ['lat', 'lon', 'rotated_pole']).

    Returns
    -------
    None
    """
    for missing_file in missing_filepaths:
        product = None
        try:
            # sse the first complete file as the template
            template_file = complete_filepaths[0]
            with nc4.Dataset(template_file, 'r') as complete_ds:
                with nc4.Dataset(missing_file, 'w', format='NETCDF4') as new_ds:
                    # copy global attributes
                    for attr_name in complete_ds.ncattrs():
                        setattr(new_ds, attr_name, getattr(complete_ds, attr_name))
                    _add_history_attr(new_ds, "Created from scratch with variables filled with NaN.")

                    # extract product and reference_time from filename
                    product = new_ds.getncattr('product')
                    filename_stem = Path(missing_file).stem  # e.g., '2017021500'
                    try:
                        reference_time = pd.to_datetime(filename_stem, format="%Y%m%d%H")  # e.g., 2017-02-15 00:00:00
                    except ValueError as ve:
                        logging.error(f"Filename '{missing_file}' does not match format '%Y%m%d%H'. Error: {ve}")
                        continue  

                    # time values to assign 
                    time_values = np.array([6, 7, 8, 9, 10, 11, 12], dtype='int32') 

                    # define the time dimension with the exact size
                    new_ds.createDimension('time', len(time_values))  

                    # retrieve the 'time' variable attributes from the template
                    template_time_var = complete_ds.variables['time']
                    time_units_updated = f"hours since {reference_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    time_calendar = getattr(template_time_var, 'calendar', 'gregorian')  # default to gregorian 

                    # create and assign the 'time' variable with integer values
                    time_var = new_ds.createVariable('time', 'i4', ('time',), fill_value=999999)  # 'i4' for int32
                    time_var[:] = time_values                                                     # assign integer time values
                    time_var.units = time_units_updated                                           # updated units
                    time_var.calendar = time_calendar
                    time_var.long_name = getattr(template_time_var, 'long_name', 'time')
                    time_var.standard_name = getattr(template_time_var, 'standard_name', 'time')
                    time_var.axis = getattr(template_time_var, 'axis', 'T')

                    # copy other dimensions (excluding 'time')
                    for dim_name, dim in complete_ds.dimensions.items():
                        if dim_name == 'time':
                            continue  # already handled
                        if dim_name not in new_ds.dimensions:
                            new_ds.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)

                    # copy variables
                    for var_name in complete_ds.variables.keys():
                        if var_name == 'time':
                            continue  # already handled

                        var = complete_ds.variables[var_name]

                        if '_FillValue' in var.ncattrs():
                            fill_value = var._FillValue
                        else:
                            # assign default fill value based on data type
                            if var.dtype.kind == 'i':
                                fill_value = 999999
                            elif var.dtype.kind == 'f':
                                fill_value = np.nan

                        var_dtype = var.dtype

                        new_var = new_ds.createVariable(var_name, var_dtype, var.dimensions, fill_value=fill_value)

                        # copy variable attributes
                        for attr_name in var.ncattrs():
                            if attr_name != '_FillValue':
                                setattr(new_var, attr_name, getattr(var, attr_name))

                        if var_name in coordinate_vars:
                            # copy coordinate data directly
                            new_var[:] = var[:]
                        else:
                            # fill with NaN or appropriate fill value
                            shape = tuple(len(complete_ds.dimensions[dim]) for dim in var.dimensions)
                            if var.dtype.kind == 'i':
                                fill = fill_value  # e.g., 999999
                            elif var.dtype.kind == 'f':
                                fill = fill_value  # e.g., np.nan
                            else:
                                fill = fill_value  # default

                            # create a filled array
                            filled_data = np.full(shape, fill, dtype=new_var.dtype)
                            new_var[:] = filled_data
                           
        except OSError as e:
            logging.error(f"Failed to open complete file '{complete_filepaths[0]}': {e}")
        except Exception as e:
            logging.error(f"An error occurred while creating file '{missing_file}': {e}")
        else:
            logging.info(f"Created new {product} file '{missing_file}' with the same variables from the complete file '{complete_filepaths[0]}' and filled with NaN.")


def read_data(RDPS_directory : str, CaPA_directory : str, tgt_file : str, start_date : datetime) -> Tuple[list, list, nc4.Dataset, nc4.Dataset, dict]:
    """
    Reads in the input netCDF files
    Parameters
    ----------
    RDPS_directory : str
        path of the rdps files.
    CaPA_directory : str
        path of the capa files.
    tgt_file : str
        name of the target file used for regridding and as template.
    start_date : str
        start date in the format of YYYYMMDD

    Returns
    -------
    RDPS_datasets : list
        list of the rdps files read as netcdf.
    CaPA_datasets : list
        list of the capa files read as netcdf.
    tgt_ds : netcdf
        netcdf object with the information of the target file used for regridding and as template.
    namefile: dict
        
    """
 
    # turn the start_date datetime object into a string
    start_date_string = start_date.strftime("%Y%m%d")
    
    # also get the string for the date following the start date
    next_date         = start_date + timedelta(days = 1)
    next_date_string  = next_date.strftime("%Y%m%d")
    
    # create empty list to store the RDPS, CaPA and the name of files
    RDPS_datasets  = []
    namefile    = {}
    CaPA_datasets  = []
    
    # list of needed files from RDPS and CaPA
    namefile['RDPS'] = [join(RDPS_directory, start_date_string + "06.nc"),
                        join(RDPS_directory, start_date_string + "12.nc"),
                        join(RDPS_directory, start_date_string + "18.nc"),
                        join(RDPS_directory, next_date_string  + "00.nc")]
    
    namefile['CaPA'] = [join(CaPA_directory, start_date_string + "18.nc"),
                        join(CaPA_directory, next_date_string + "00.nc"),
                        join(CaPA_directory, next_date_string + "06.nc"),
                        join(CaPA_directory, next_date_string + "12.nc")]
            
    # Check if all RDPS files are present
    for file in namefile['RDPS']: 
        try:
            RDPS_datasets.append(nc4.Dataset(file, 'r'))
            # logging.info(f"RDPS file {file} is present --> OK")
        except:
            sys.stderr.write(f"PROBLEM opening (or non existing) RDPS file: {file}")
        
    
    # Check if all CaPA files are present
    for file in namefile['CaPA']: 
        try:
            CaPA_datasets.append(nc4.Dataset(file, 'r'))    
            # logging.info(f"CaPA file {file} is present --> OK")
        except:
            sys.stderr.write(f"PROBLEM opening (or non existing) CaPA file: {file}")

    # read the target file 
    tgt_ds = nc4.Dataset(tgt_file, 'r')


    return RDPS_datasets, CaPA_datasets, tgt_ds, namefile


def check_time(RDPS_files : list, required_time : list, nameprodfile : dict) -> Tuple[list, list]:
    """
    Checks to make sure the RDPS files have the required 7 hours of data, 
    and that each file contains the same forecast horizons
    
    Parameters
    ----------
    RDPS_files : list
         list of netcdf4 datasets (RDPS).
    required_time : list
        list of the required time steps (lead time 6-12)
    nameprodfile : dict
        information on the name of files

    Returns
    -------
    start_index : list
        indicate the index for the corresponding lead time (start)
    end_index : list
        indicate the index for the corresponding lead time (end)

    """
    compt = 0
    start_index = []
    end_index   = []
    
    for file in RDPS_files:
        # name of the fie
        name_rdps = nameprodfile['RDPS'][compt]
        
        # boolean list check if the required time window is the files
        lst = [x in file.variables["time"][:] for x in required_time]
        
        if(all(lst)):
            pass
        else:
            missinghours = list(np.where(~np.array(lst))[0])
            sys.exit(f"""RDPS file {name_rdps} must contain at least {len(required_time)} hours
                     from {required_time[0]} to {required_time[-1]} forecast time window - 
                     the following hour(s) {missinghours} is(are) missing""")

        compt       = compt + 1
        
        start_index.append(int(np.where(file.variables["time"][:]==required_time[1])[0]))
        end_index.append(int(np.where(file.variables["time"][:]==required_time[-1])[0]) + 1)
        
    return start_index, end_index


def create_netCDF(tgt_ds : nc4.Dataset, outfile_nc : str) -> nc4.Dataset:
    """
    Creates a new netCDF file for output.

    Parameters
    ----------
    tgt_ds : netcdf dataset
        target file used for regridding and as template.
    outfile_nc : str
        name of the output netcdf file.

    Returns
    -------
    ncid : nc id
        identifiant of the netcdf file.

    """

    # create new netCDF file for output
    ncid             = nc4.Dataset(outfile_nc, "w", format="NETCDF4")
    
    # create dimensions for netCDF file
    for name, dimension in tgt_ds.dimensions.items(): 
        if name == 'time':
            ncid.createDimension(name, 24) # 24 hours
        else:
            ncid.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
        
    # set file-level attributes
    new_attrs = {
        'product': 'RDPS_CaPA',
        'Remarks' : "Variable names are following the convention <Product>_<Type:A=Analysis,P=Prediction>_<ECCC name>_<Level/Tile/Category>. Variables with level \'10000\' are at surface level. The height [m] of variables with level \'0XXXX\' needs to be inferrred using the corresponding fields of geopotential height (GZ_0XXXX-GZ_10000). The variables UUC, VVC, UVC, and WDC are not modelled but inferred from UU and VV for convenience of the users. Precipitation (PR) is reported as 6-hr accumulations for CaPA_fine and CaPA_coarse. Precipitation (PR) are accumulations since beginning of the forecast for GEPS, GDPS, REPS, RDPS, HRDPS, and CaLDAS. The re-analysis product CaSR (_v1, _v2, v2.1, v3.1) contains two variables for precipitation: \'P_PR_SFC\' is the model precipitation (trial field used by CaPA) and \'A_PR_SFC\' is precipitations adjusted with CaPA 24h precipitation. Please be aware that the baseflow \'O1\' of the current version of WCPS is not reliable during the spring melt period. ORRC is not a product available directly on CaSPar. It is created by combining RDPS and CaPA outputs."
    }
    ncid.setncatts(new_attrs)
    
    return ncid


def netCDF_variable_assignment(ncid : nc4.Dataset, outfile_nc : str, var_list : list, RDPS_datasets : list, tgt_ds : nc4.Dataset, start_index : list, end_index : list) -> None:
    """
    Creates the variables in the output NetCDF file. 

    Parameters
    ----------
    ncid : nc id
        identifiant of the netcdf file.
    outfile_nc : str    
        name of the output netcdf file.
    var_list : list
        list of variables to be added in the file output.
    RDPS_datasets : list
        list of RDPS netcdf datasets.
    tgt_ds : netcdf dataset
        target file used for regridding and as template.
    start_index : list
        list of the start index (to take the right time step - start index).
    end_index : list
        list of the start index (to take the right time step - end index).

    Returns
    -------
    None.

    """
    # outfile_nc example: 'RDPS_CaPA_2015030812.nc'
    ref_time = pd.to_datetime(Path(outfile_nc).stem.split('_')[-1], format="%Y%m%d%H")

    # _FillValue is the only attribute that can't be added after the variable is created
    fill_value = tgt_ds['time'].__dict__.get('_FillValue')
    if not (isinstance(fill_value, (int, np.integer))):
        fill_value = None      # drop it if it's not an int

    # passing an empty dict will use netcdf's default fill value
    # creating a 32-bit integer variable for time so the FillValue must be an int
    kwargs = {'fill_value': fill_value} if fill_value is not None else {}
    times  = ncid.createVariable('time', 'i4', ('time',), **kwargs)

    for attr, value in tgt_ds['time'].__dict__.items():
        if attr in ('units', '_FillValue'):
            # replace units, skip _FillValue (already handled)
            if attr == 'units':
                times.setncattr(attr, f"hours since {ref_time:%Y-%m-%d %H:%M:%S}")
        else:
            times.setncattr(attr, value)
    times[:] = np.arange(1, 25, 1)         
    # this will convert integer offsets into actual datetimes based on the combination of numeric values + reference units

    # create variable holding location of the rotated pole
    pole = ncid.createVariable('rotated_pole', 'f4')
    pole.setncatts(tgt_ds['rotated_pole'].__dict__) 
    
    # create spatial variables and attributes
    for latlon in ('lat','lon'):
        rlatlon          = 'r'+latlon
        ncid.createVariable(rlatlon,'f4',(rlatlon,))
        
        ncid[rlatlon].setncatts(tgt_ds[rlatlon].__dict__) 
        ncid[rlatlon][:] = tgt_ds[rlatlon][:]
        
        ncid.createVariable(latlon,'f4',('rlat','rlon'))
        ncid[latlon].setncatts(tgt_ds[latlon].__dict__) 
        ncid[latlon][:] = tgt_ds[latlon][:]

    # create all other variables (except precipitation analysis)
    # loop through each variable name from input RDPS_files
    for var in RDPS_datasets[0].variables:
        if var in var_list:       
            _variable_data(ncid, var, RDPS_datasets, start_index, end_index)


def _variable_data(ncid : nc4.Dataset, key_var : str, RDPS_datasets : list, start_index : list, end_index : list) -> None:
    """
    Assigns the data to the new variable in the output NetCDF file.
    If the variable is precipitation, it forward-fills the data.
    
    Parameters
    ----------
    ncid : nc id
        identifiant of the netcdf file.
    key_var : str
        name of the variable.
    RDPS_datasets : list
        list of RDPS netcdf datasets.
    start_index : list
        list of the start index.
    end_index : list
        list of the end index.
    
    Returns
    -------
    None.
    """

    # collect source metadata
    source_var = RDPS_datasets[0].variables[key_var]
    var_attrs = {attr_name: source_var.getncattr(attr_name) for attr_name in source_var.ncattrs()}

    # create destination variable
    var_name = "RDPS_CaPA_" + key_var[5:]
    ncvar = ncid.createVariable(var_name, 'f4', ('time', 'rlat', 'rlon'))

    # load, concat
    arrays = [
        xr.DataArray(ds.variables[key_var][s:e], dims=("time", "rlat", "rlon"))
        for ds, s, e in zip(RDPS_datasets, start_index, end_index)
    ]
    data_da = xr.concat(arrays, dim="time")

    # if precipitation, forward fill missing hours
    # during preprocessing the data is already forward filled 
    # but there could be cases where the first time step is missing in an individual file
    if key_var == "RDPS_P_PR_SFC":
        data_da = data_da.ffill("time")

    # write the data to the new variable
    ncvar[:] = data_da.data.astype("float32", copy=False)

    # clean and copy attributes
    var_attrs_clean = {}
    for attr_name, attr_value in var_attrs.items():
        # skip _FillValue attribute, already handled by createVariable
        if attr_name == '_FillValue':
            continue

        # ensure attributes are of a type that netCDF can handle
        if isinstance(attr_value, (str, float, int, np.number)):
            var_attrs_clean[attr_name] = attr_value
        else:
            sys.exit(f"Attribute '{attr_name}' of variable '{key_var}' is of type '{type(attr_value)}' which is not supported by netCDF.")

    # set attributes
    ncvar.setncatts(var_attrs_clean)

       
def do_A_PR_SFC(ncid : nc4.Dataset, RDPS_datasets : list, CaPA_datasets : list, tgt_ds : nc4.Dataset) -> None:
    """
    Create the "Analysis: Quantity of precipitation" variable in the output netCDF file
    Adjusts the hourly RDPS values using a scaling factor based on the ratio of CaPA to RDPS totals
    to ensures that forecasted precipitation data (RDPS) better matches observed values (CaPA). 

    Parameters
    ----------
    ncid : nc id
        identifiant of the netcdf file.
    RDPS_datasets : list
        list of RDPS netcdf datasets.
    CaPA_datasets : list
        list of CaPA netcdf datasets.
    tgt_ds : netcdf dataset
        CaSR file used as a template for the output file. It is used to get the attributes of the variables.

    Returns
    -------
    None.
    """

    # read RDPS precipitation and forward fill missing hours
    rdps_data = []
    for f in RDPS_datasets:
        da = xr.DataArray(f.variables["RDPS_P_PR_SFC"][:], dims=("time", "rlat", "rlon")).ffill("time")                       
        rdps_data.append(da.values.astype("float32", copy=False))

    # read CaPA 6h accumulated precipitation
    number_of_hours = 6
    number_of_files = len(CaPA_datasets)

    capa_data = []
    for n in range(number_of_files):
        capa_pr = CaPA_datasets[n].variables['CaPA_coarse_A_PR_SFC'][:][0]
        capa_data.append(capa_pr)

    # set empty array for the hourly precipitation data with the same shape as the RDPS data
    precip_data_hourly = [
        np.empty_like(item[:number_of_hours])  # (6, rlat, rlon)
        for item in rdps_data
    ]

    # convert cumulative RDPS fields to hourly increments
    for n in range(number_of_files):
        for i in range(number_of_hours):
            precip_data_hourly[n][i] = rdps_data[n][i + 1] - rdps_data[n][i]

    # get RDPS total precip for the same 6h window as CaPA by summing the hourly increments
    rdps_precip_daily_sums = []
    for item in precip_data_hourly:
        rdps_precip_daily_sums.append(np.sum(item, axis = 0))

    # scale RDPS hourly totals so their six-hour sum matches CaPA
    for n in range(number_of_files):
        scale       = capa_data[n]/rdps_precip_daily_sums[n]
        capa_hr_cst = capa_data[n]/number_of_hours
        
        for t in range(number_of_hours):
            # if RDPS forecasted no precip, spread CaPA’s 6h total evenly over the six hours
            # if RDPS forecasted precip, scale the RDPS hourly total by the ratio of CaPAtotal / RDPStotal
            precip_data_hourly[n][t] = np.where(rdps_precip_daily_sums[n]>1e-6, scale*precip_data_hourly[n][t], capa_hr_cst)    
               
    precip_data = np.array(precip_data_hourly)
    precip_data = np.concatenate(precip_data, axis = 0)
    
    # create precipitation variable
    ncid.createVariable('RDPS_CaPA_A_PR_SFC', 'f4', ('time', 'rlat', 'rlon'))
    
    name_product=tgt_ds.__dict__['product']  
    name_product_PR=str(name_product)+'_A_PR0_SFC'
       
    ncid['RDPS_CaPA_A_PR_SFC'].setncatts(tgt_ds[name_product_PR].__dict__) 
    ncid['RDPS_CaPA_A_PR_SFC'][:] = precip_data
    
    return(precip_data)


def create_orrc(RDPS_regridded_dir : str, CaPA_regridded_dir : str, tgt_file : str, start_date : str, outfile_nc : str, var_list : list, required_time_window : list, analysis_bool : bool) -> None:
    """
    Creates a new netCDF file for output.

    Parameters
    ----------
    RDPS_regridded_dir : str
        path of the regridded rdps files.
    CaPA_regridded_dir : str
        path of the regridded capa files.
    tgt_file : str
        name of the target file used for regridding and as template.
    start_date : str
        start date in the format of YYYYMMDD
    outfile_nc : str
        name of the output netcdf file.
    var_list : list
        list of variables to be added in the file output.
    required_time_window : list
        list of the required time steps (lead time 6-12)
    analysis_bool : bool
        boolean to create the analysis variable.
    
    Returns
    -------
    None.
    """
    RDPS_datasets, CaPA_datasets, tgt_ds, namefile = read_data(RDPS_regridded_dir, CaPA_regridded_dir, tgt_file, start_date)

    # check to see if input RDPS files contain the required 7 hours and RDPS files contain the same forecast horizons
    # and get the start and end index of the required time window
    start_index, end_index = check_time(RDPS_datasets, required_time_window, namefile)

    # netCDF will be created based on the CaSR v3.1 grid 
    ncid = create_netCDF(tgt_ds, outfile_nc)

    # create variables in output netCDF file
    netCDF_variable_assignment(ncid, outfile_nc, var_list, RDPS_datasets, tgt_ds, start_index, end_index)

    if analysis_bool:
        # create "Analysis: Quantity of precipitation" variable
        do_A_PR_SFC(ncid, RDPS_datasets, CaPA_datasets, tgt_ds)

    ncid.close()


def split_ds(ds : xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:  
    """
    Splits the dataset into two datasets: one with complete years and one with partial years.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset to be split.
    Returns 
    -------
    ds_cmp : xr.Dataset or None
        A dataset containing only complete years. Returns None if there are no complete years.

    """ 
    # return zarr with all complete years and zarr with partial
    nday = 31 if ds.time.dt.calendar != '360_day' else 30
    if ds.time[-1].dt.month == 12 and ds.time[-1].dt.day == nday:
        last_year = ds.time[-1].dt.year.values
    else:
        last_year = ds.time[-1].dt.year.values - 1
    
    ds_cmp = ds.sel(time=(ds.time.dt.year<=last_year)) if any(ds.time.dt.year<=last_year) else None
        
    ds_prt = ds.sel(time=(ds.time.dt.year>last_year)) if any(ds.time.dt.year>last_year) else None
        
    return ds_cmp, ds_prt


def zarr_to_zarr_zip(dsout : xr.Dataset, outpath : Path, variable : xr.DataArray, timechunk : int, fmt : str, working_dir : Path, dask_kwargs: dict) -> None:
    """
    Saves the dataset to a zarr file and then zips it.

    Parameters
    ----------
    dsout : xr.Dataset
        The dataset to be saved.
    outpath : Path
        The path where the output zarr.zip file will be saved.
    variable : xr.DataArray
        The variable in the dataset that is being processed (used for rechunking).
    timechunk : int 
        The chunk size for the time dimension when rechunking the dataset for saving.   
    fmt : str
        The format of the output file (e.g., 'zarr').
    working_dir : Path  
        The working directory where the temporary zarr file will be created before zipping.
    dask_kwargs : dict
        A dictionary of keyword arguments to be passed to the Dask Client when saving the dataset.
    
    Returns
    -------
    None.
    """
    with TemporaryDirectory(prefix=f"{working_dir.as_posix()}/") as tmpdir:
        tmpfile = Path(tmpdir).joinpath(outpath.name)
        chunks = {d: 50 for d in dsout[variable].dims if d != 'time'}
        chunks['time'] = timechunk
        dsout = xs.io.rechunk_for_saving(dsout, rechunk=chunks)
        for attr in [d for d in dsout.attrs if d.startswith('cat:')]:
            del dsout.attrs[attr]
        with Client(**dask_kwargs):
            xs.io.save_to_zarr(dsout, filename=tmpfile)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        if fmt == 'zarr':
            xs.io.zip_directory(root=tmpfile, zipfile=tmpfile.with_suffix(".zarr.zip"), delete=True)
        shutil.move(tmpfile.with_suffix(".zarr.zip"), outpath.with_suffix(".zarr.zip"))

    
def format_user(obj : Union[dict, list, str], user : str) -> Union[dict, list, str]:
    """ 
    Recursively formats nested structures (dicts, lists, strings) by replacing '{user}' with the provided user. 

    Parameters
    ----------
    obj : dict, list, or str
        The object to be formatted. Can be a dictionary, list, or string.
    user : str
        The string to replace '{user}' with in the input object.

    Returns
    -------
    The formatted object with '{user}' replaced by the provided user string. The structure of the input object is preserved.
    
    """
    if isinstance(obj, dict):
        return {k: format_user(v, user) for k, v in obj.items()}
    if isinstance(obj, list):
        return [format_user(v, user) for v in obj]
    if isinstance(obj, str):
        return obj.replace("{user}", user)
    return obj


def _remove_all_suffixes(p: Path) -> Path:
    """
    Removes all file extensions (suffixes) from a pathlib.Path object.

    Parameters
    ----------
    p : Path
        The input Path object from which to remove all suffixes.

    Returns 
    -------
    Path
        A new Path object with all suffixes removed.
    """
    # Convert to a PurePath to work with virtual paths efficiently
    p = Path(p.name) # Start just with the filename part to avoid path issues
    while p.suffix:
        p = p.with_suffix('')
    return p


def get_datelen_format(filepath : Path) -> Tuple[int, str]:
    """
    Determines the length of the date string in the filename and returns the corresponding date format string for parsing.
    
    Parameters
    ----------
    filepath : Path
        The path to the file whose date format is to be determined.
    Returns
    -------
    tuple
        A tuple containing the length of the date string and the corresponding date format string for parsing.
    """
    datelen = len(_remove_all_suffixes(filepath).name.split('_')[-1].split('-')[0])
    if datelen == 4:
        date_fmt='%Y'
    elif datelen == 6:
        date_fmt='%Y%m'
    elif datelen == 8:
        date_fmt='%Y%m%d'
    else:
        raise ValueError("Unexpected date length in filename.")

    return datelen, date_fmt


def _extract(ds: xr.Dataset) -> list:
    """
    Extracts the start and end times from an xarray.Dataset object.
    
    Parameters
    ----------
    ds : xr.Dataset
        The input xarray.Dataset from which to extract the start and end times.

    Returns
    -------
    list of datetime
        A list containing the start and end times extracted from the dataset.
    """
    t = pd.to_datetime(ds["time"].values)
    start = t[0].to_pydatetime()
    end   = t[-1].to_pydatetime()
    return [start, end]


def _get_start_end_times(filepath : Path) -> list:
    """
    Extracts the start and end times from a netCDF or zarr file.
    
    Parameters
    ----------
    filepath : Path
        Path to the netCDF or zarr file.
    
    Returns
    -------
    list of datetime
        A list containing the start and end times extracted from the file.

    """
    try:
        with xr.open_dataset(filepath, decode_times=True) as ds:
            return _extract(ds)
    except Exception:
        pass

    ds = xr.open_zarr(filepath) 
    try:
        return _extract(ds)
    finally:
        ds.close()


def remove_partial_zarrs(infolder : Path) -> Union[list, None]:
    """ 
    Remove any year to date files that are covered by complete year files. 

    Parameters
    ----------
    infolder : Path
        Path to the folder containing the zarr.zip files.
    Returns
    -------
    list of Path or None
        List of Path objects representing the zarr.zip files to be removed, or None if no files need to be removed.
    """
    all_zarrs = sorted(list(infolder.rglob("*zarr.zip")))
    prtl_zarrs = [p for p in all_zarrs if '-' not in _remove_all_suffixes(p).name.split('_')[-1] or get_datelen_format(p)[0] != 4]
    remove_list = []
    for pz in prtl_zarrs:
        pdates = _get_start_end_times(pz)
        edates = [_get_start_end_times(p) for p in all_zarrs if p != pz] 
        if any([pdates[0] >= edate[0] and pdates[1] <= edate[1] for edate in edates]):
            remove_list.append(pz)
    if remove_list:
        return remove_list
    else:
        return None
    

def mask_data(ds : xr.Dataset, variable : str, mask : dict) -> xr.Dataset:
    """
    Mask the dataset based on the provided mask information. 
    If the variable is in the mask and the time range overlaps, the data will be masked (set to NaN) for the specified time range.
    
    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be masked.
    variable : str
        The variable name to check against the mask.
    mask : dict
        A dictionary containing the mask information with keys "variables", "start_time", and "end_time".

    Returns
    -------
    xarray.Dataset
        The masked dataset.
    """
    
    if not mask:
        return ds

    mask_variables = mask.get("variables", [])
    mask_start_time = mask.get("start_time")
    mask_end_time = mask.get("end_time")

    if variable not in mask_variables:
        return ds
    if mask_start_time is None or mask_end_time is None:
        return ds
    if "time" not in ds.coords:
        return ds

    mask_start = np.datetime64(mask_start_time)
    mask_end = np.datetime64(mask_end_time)

    ds_start = np.datetime64(ds.time.min().values)
    ds_end = np.datetime64(ds.time.max().values)

    # skip if there is no overlap at all
    if ds_end < mask_start or ds_start > mask_end:
        return ds

    return ds.where((ds.time < mask_start) | (ds.time > mask_end))