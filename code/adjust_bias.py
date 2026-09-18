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
    staging_dir.mkdir(parents=True, exist_ok=True)

    reconstruction_root = staging_dir / CONFIG["main"]["reconstruction_dir"]

    overwrite = CONFIG["biasadjust"]["overwrite"]
    frequency = CONFIG["biasadjust"]["frequency"]
    domain = CONFIG["biasadjust"]["domain"]

    logging.info(f"Overwrite option is set to {overwrite}.")

    ba_product = dict(CONFIG["biasadjust"]["biasadj_product"])
    if version not in ba_product:
        raise ValueError(
            f"Version {version} not found in biasadjust.biasadj_product in config file."
        )
    
    ba_source = ba_product[version]["source"]
    ba_version = ba_product[version]["version"]
    ba_url = ba_product[version]["url"]
    apply_on = CONFIG["biasadjust"]["apply_on"]

    if not ba_url:
        raise ValueError(
            f"biasadjust.biasadj_product.{version}.url is empty, but it is required to train the bias adjustment."
        )

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

    ds_ba_train_all = xr.open_dataset(ba_url, engine="netcdf4", chunks=dict(time=(365*4)+1, rlat=50, rlon=50))
    ds_ref_all = xr.open_dataset(ref_url, engine="netcdf4", chunks=dict(time=(365*4)+1, rlat=50, rlon=50))

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

            if source_attr == ba_source and version_attr == ba_version:
                mod_files.append(file)

            ds.close()
        
        if var not in ds_ba_train_all.data_vars:
            raise KeyError(f"Variable '{var}' not found in the bias adjustment dataset {ba_source} {ba_version} at {ba_url}.")
        if var not in ds_ref_all.data_vars:
            raise KeyError(f"Variable '{var}' not found in the reference dataset {ref_source} {ref_version} at {ref_url}.")

        ds_ba_train = ds_ba_train_all[[var]]
        ds_ref = ds_ref_all[[var]]

        ba_train_start = pd.Timestamp(ds_ba_train.time.min().values)
        ba_train_end = pd.Timestamp(ds_ba_train.time.max().values)
        ref_start = pd.Timestamp(ds_ref.time.min().values)
        ref_end = pd.Timestamp(ds_ref.time.max().values)

        if config_end < ba_train_start or config_start > ba_train_end:
            logging.warning(
                f"Requested ref_period [{config_start}, {config_end}] does not overlap "
                f"with bias adjustment {ba_source} {ba_version} data for var='{var}' [{ba_train_start}, {ba_train_end}]."
            )

        if config_end < ref_start or config_start > ref_end:
            logging.warning(
                f"Requested ref_period [{config_start}, {config_end}] does not overlap "
                f"with reference {ref_source} {ref_version} data for var='{var}' [{ref_start}, {ref_end}]."
            )

        train_start = max(config_start, ba_train_start, ref_start)
        train_end = min(config_end, ba_train_end, ref_end)

        if train_start > train_end:
            raise ValueError(
                f"No usable training overlap for var='{var}'. "
                f"Requested=[{config_start}, {config_end}], "
                f"{ba_source} {ba_version}=[{ba_train_start}, {ba_train_end}], "
                f"reference {ref_source} {ref_version}=[{ref_start}, {ref_end}]"
            )

        ds_ba_per = ds_ba_train.sel(time=slice(train_start, train_end))
        ds_ref_per = ds_ref.sel(time=slice(train_start, train_end))

        # apply bias adjustment to either the data from the URL or the local staging files
        if apply_on == "url":
            ds_apply = ds_ba_train

        elif apply_on == "staging":
            if len(mod_files) == 0:
                raise FileNotFoundError(
                    f"No local staging files found for var='{var}', source='{ba_source}', version='{ba_version}', frequency='{frequency}', domain='{domain}'."
                )

            local_datasets = [xr.open_zarr(file, consolidated=True, chunks={})[[var]] for file in sorted(mod_files)]

            if len(local_datasets) == 1:
                ds_apply = local_datasets[0]
            else:
                ds_apply = xr.concat(
                    local_datasets,
                    dim="time",
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                    combine_attrs="override",
                ).sortby("time")

        else:
            raise ValueError(
                f"Unsupported biasadjust.apply_on: {apply_on}. Expected 'url' or 'staging'."
            )

        # create bias_adjust_reference string for global attributes
        bias_adjust_reference = (
            f"{ref_source} {ref_version.replace('.', '')} "
            f"{train_start.year}-{train_end.year}"
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
            "frequency": frequency,
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
        var_mod_per = ds_ba_per[var].drop_vars(['rotated_pole'], errors="ignore").chunk({'time': -1})
        var_mod = ds_apply[var].drop_vars(['rotated_pole'], errors="ignore").chunk({'time': -1})

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
        if "rotated_pole" in ds_apply.coords:
            ds_adj = ds_adj.assign_coords(rotated_pole=ds_apply.rotated_pole)
    
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
                outzip = outpath.with_suffix('.zarr.zip')

                # update processing_level attr before saving; keeps the original path structure
                dsout.attrs["processing_level"] = "biasadjusted"

                if not outzip.exists() or overwrite:
                    if outzip.exists() and overwrite:
                        outzip.unlink()
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                    zarr_to_zarr_zip(dsout, outpath, var, timechunk, fmt, working_dir, dask_kwargs)
                    logging.info(f"Successfully created new file {outzip}.")
                        
                # remove any partial zarrs already covered
                remove_list = remove_partial_zarrs(outpath.parent)
                if remove_list:
                    for p in remove_list:
                        p.unlink()
                        logging.info(f"Removed partial zarr file already covered: {p}")


if __name__ == "__main__":

    config = str(Path(__file__).parent.joinpath('data', 'config_biasadj.yml'))  

    config_orrc_path = Path(__file__).parent.joinpath('data', 'config_orrc.yml')
    with open(config_orrc_path, 'r') as f:
        config_orrc = yaml.safe_load(f)   

    for version in config_orrc['Settings']['versions']:
        main(config, version)