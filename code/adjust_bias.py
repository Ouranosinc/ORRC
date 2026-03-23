#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script bias adjusts ORRC data using the xsdba library on a reference dataset over a specified reference period.

"""
import yaml
import logging
import os
import sys

from pathlib import Path
import xarray as xr
import xscen as xs
import pandas as pd
import xsdba

from xscen.config import CONFIG, load_config

from utils import split_ds, zarr_to_zarr_zip, format_user, get_datelen_format, remove_partial_zarrs

def main(config, version):

    user = os.getlogin()

    load_config(config, verbose=True)

    logfile = Path(CONFIG['main']['logging_dir'].format(user=user)).joinpath(version.replace('.', ''), 'adjust_bias.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    home = Path("~").expanduser()
    dask_dir = home.joinpath("my_dir", "tmp_orrc", "dask")
    dask_dir.mkdir(parents=True, exist_ok=True)
    dask_kwargs = dict(
        n_workers=6,
        threads_per_worker=6, 
        memory_limit="12GB",
        dashboard_address=8991,
        local_directory=dask_dir,
        silence_logs=logging.ERROR,
    )

    working_dir = Path(CONFIG["main"]["working_dir"].format(user=user))
    working_dir.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(CONFIG["main"]["staging_dir"].format(user=user))
    reconstruction_root = staging_dir / CONFIG["main"]["reconstruction_dir"]

    overwrite = CONFIG["biasadjust"]["overwrite"]
    frequency = CONFIG["biasadjust"]["frequency"]
    domain = CONFIG["biasadjust"]["domain"]

    logging.info(f"Overwrite option is set to {overwrite}.")

    ref_product = dict(CONFIG["biasadjust"]["ref_product"])
    if version not in ref_product:
        raise ValueError(
            f"Version {version} not found in biasadjust.ref_product in config file."
        )

    ref_source = ref_product[version]["source"]
    ref_version = ref_product[version]["version"]
    ref_url = ref_product[version]["url"]

    ref_period = dict(CONFIG["biasadjust"]["ref_period"])
    if version not in ref_period:
        raise ValueError(
            f"Version {version} not found in biasadjust.ref_period in config file."
        )

    config_start = pd.Timestamp(ref_period[version]["start_date"])
    config_end = pd.Timestamp(ref_period[version]["end_date"])

    var_specs = dict(CONFIG["biasadjust"]["variables"])

    ds_ref_all = xr.open_dataset(ref_url, engine="netcdf4")

    all_zarrzip = list((reconstruction_root / domain).rglob("*.zarr.zip"))

    for var in var_specs:
        mod_files = []

        # find the specified files 
        for file in all_zarrzip:
            ds = xr.open_zarr(file, consolidated=True, chunks={})

            if var not in ds.data_vars:
                ds.close()
                continue

            source_attr = ds.attrs.get("source")
            version_attr = ds.attrs.get("version")
            frequency_attr = ds.attrs.get("frequency")
            domain_attr = ds.attrs.get("domain")

            if frequency_attr != frequency or domain_attr != domain:
                ds.close()
                continue

            if source_attr == "ORRC" and version_attr == version:
                mod_files.append(file)

            ds.close()
        
        if len(mod_files) == 0:
            raise FileNotFoundError(
                f"No model files found for var='{var}', source='ORRC', version='{version}', "
                f"frequency='{frequency}', domain='{domain}'."
            )
        
        # open the dataset
        mod_datasets = [xr.open_zarr(file, consolidated=True, chunks={}) for file in sorted(mod_files)]

        if len(mod_datasets) == 1:
            ds_mod = mod_datasets[0]
        else:
            ds_mod = xr.concat(
                mod_datasets,
                dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
                combine_attrs="override",
            ).sortby("time")

        ds_ref = ds_ref_all[[var]]

        mod_start = pd.Timestamp(ds_mod.time.min().values)
        mod_end = pd.Timestamp(ds_mod.time.max().values)

        ref_start = pd.Timestamp(ds_ref.time.min().values)
        ref_end = pd.Timestamp(ds_ref.time.max().values)

        if config_end < mod_start or config_start > mod_end:
            logging.warning(
                f"Requested ref_period [{config_start}, {config_end}] does not overlap "
                f"with ORRC data for var='{var}' [{mod_start}, {mod_end}]."
            )

        if config_end < ref_start or config_start > ref_end:
            logging.warning(
                f"Requested ref_period [{config_start}, {config_end}] does not overlap "
                f"with reference {ref_source} {ref_version} data for var='{var}' [{ref_start}, {ref_end}]."
            )

        start_date = max(config_start, mod_start, ref_start)
        end_date = min(config_end, mod_end, ref_end)

        if start_date > end_date:
            raise ValueError(
                f"No usable overlap for var='{var}'. "
                f"Requested=[{config_start}, {config_end}], "
                f"ORRC=[{mod_start}, {mod_end}], "
                f"reference {ref_source} {ref_version}=[{ref_start}, {ref_end}]"
            )

        ds_mod_per = ds_mod.sel(time=slice(start_date, end_date))
        ds_ref_per = ds_ref.sel(time=slice(start_date, end_date))

        # create bias_adjust_reference string for global attributes
        bias_adjust_reference = (
            f"{ref_source} {ref_version.replace('.', '')} "
            f"{start_date.year}-{end_date.year}"
        )

        # get bias adj specs for this variable
        training_args = var_specs[var]["training_args"]
        method = training_args.get("method")
        kind = training_args.get("kind")

        grouper = training_args.get("grouper") or {}

        group = grouper.get("group")
        window = grouper.get("window")
        group_abbr = grouper.get("group_abbr")

        if grouper:
            bias_adjust_method = f"{method} {group_abbr}+{window}"
        else:
            bias_adjust_method = method

        global_attrs = {
            **CONFIG["attrs"],
            "bias_adjust_project": f"ORRC-a-{version.replace('.', '')}",
            "bias_adjust_reference": bias_adjust_reference,
            "bias_adjust_method": bias_adjust_method,
            "version": version,
        }
        ds_adj = xr.Dataset(attrs=global_attrs)

        logging.info(
            f"Bias adjustment for ORRC {version} variable '{var}' using method '{bias_adjust_method}', reference '{bias_adjust_reference}', and kind '{kind}'."
        )

        # prepare data: # xsdba requires time to be one contiguous chunk and no rotated_pole coord
        var_ref_per = ds_ref_per[var].drop_vars(['rotated_pole'], errors="ignore").chunk({'time': -1}) 
        var_mod_per = ds_mod_per[var].drop_vars(['rotated_pole'], errors="ignore").chunk({'time': -1})
        var_mod = ds_mod[var].drop_vars(['rotated_pole'], errors="ignore").chunk({'time': -1})

        # bias adjustment
        xsdba_method = getattr(xsdba, method)
        train_kwargs = {}
        if "grouper" in training_args:
            g = training_args["grouper"]
            train_kwargs["group"] = xsdba.Grouper(g["group"], window=g.get("window"))
        if kind is not None:
            train_kwargs["kind"] = kind

        trained_obj = xsdba_method.train(var_ref_per, var_mod_per, **train_kwargs)

        adj_var = trained_obj.adjust(var_mod)
        ds_adj[var] = adj_var
        # copy back rotated_pole
        if "rotated_pole" in ds_mod.coords:
            ds_adj = ds_adj.assign_coords(rotated_pole=ds_mod.rotated_pole)
    
        fmt = "zarr"
        facets = {
            'version': version.replace(".", ""),
            }
        # rechunk for saving 
        if ds_adj.attrs['frequency'] == 'day':
            ndays = 360 if ds_adj.time.dt.calendar == '360_day' else 365
            timechunk = ndays * 4
            if ds_adj.time.dt.calendar not in ['noleap', '365_day']:
                timechunk += 1

        ds_adj = ds_adj.chunk({'rlat': 50, 'rlon': 50})

        for dsout in split_ds(ds_adj):
            if dsout is not None:
                date_start = dsout.time.min().dt.strftime('%Y%m%d').item()
                outpath = xs.catutils.build_path(dsout, root=staging_dir, format=fmt, **facets)

                if not outpath.with_suffix('.zarr.zip').exists():
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                    zarr_to_zarr_zip(dsout, outpath, var, timechunk, fmt, working_dir, dask_kwargs)
                    logging.info(f"Successfully created new file {outpath.with_suffix('.zarr.zip')}.")
                else:
                    # check if there is an existing file with the same start year but fewer time steps
                    datelen, date_fmt = get_datelen_format(outpath)
                    existing_zarrzip = list(outpath.parent.glob(f"{var}_*_{dsout.time.min().dt.strftime(date_fmt).item()}-*.zarr.zip"))
                    if existing_zarrzip:
                        assert len(existing_zarrzip) == 1, f"Multiple zarr.zip files with the same start date in {outpath.parent}. Need to investigate."
                        existing_ds = xr.open_zarr(existing_zarrzip[0], consolidated=True, chunks={})
                        # verify that the existing file has the same start date
                        if "time" in dsout.dims and "time" in existing_ds.dims and date_start == existing_ds.time.min().dt.strftime('%Y%m%d').item():
                            if len(dsout.time) > len(existing_ds.time):
                                try:
                                    os.remove(existing_zarrzip[0])
                                    zarr_to_zarr_zip(dsout, outpath, var, timechunk, fmt, working_dir, dask_kwargs)
                                    logging.info(f"{outpath.with_suffix('.zarr.zip')} is longer than existing {existing_zarrzip[0]}; overwriting.")
                                except Exception as e:
                                    logging.error(f"Error removing existing file {existing_zarrzip[0]}: {e}")
                                    sys.exit("Unable to remove existing file. Exiting.")
                            elif len(dsout.time) == len(existing_ds.time):
                                if overwrite:
                                    try:
                                        os.remove(existing_zarrzip[0])
                                        zarr_to_zarr_zip(dsout, outpath, var, timechunk, fmt, working_dir, dask_kwargs)
                                        logging.info(f"{outpath.with_suffix('.zarr.zip')} has same length as existing {existing_zarrzip[0]} and overwrite=True; overwriting.")
                                    except Exception as e:
                                        logging.error(f"Error removing existing file {existing_zarrzip[0]}: {e}")
                                        sys.exit("Unable to remove existing file. Exiting.")
                                else:
                                    logging.info(f"Skipping {outpath.with_suffix('.zarr.zip')} as it has the same number of time points as existing file and overwrite=False.")
                                    continue
                            else: # this scenario should not happen, but added for safety
                                logging.warning(f"Skipping {outpath.with_suffix('.zarr.zip')} as it has fewer time points than existing file. Need to investigate.")
                                continue
                        existing_ds.close()
                        
                # remove any partial zarrs covered by complete years; should only occur for first run of the year
                remove_list = remove_partial_zarrs(outpath.parent)
                if remove_list:
                    for p in remove_list:
                        p.unlink()
                        logging.info(f"Removed partial zarr file covered by complete year: {p}")

if __name__ == "__main__":

    config = str(Path(__file__).parent.joinpath('data', 'config_biasadj.yml'))  

    config_orrc_path = Path(__file__).parent.joinpath('data', 'config_orrc.yml')
    with open(config_orrc_path, 'r') as f:
        config_orrc = yaml.safe_load(f)   

    for version in config_orrc['Settings']['versions']:
        main(config, version)