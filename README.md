# ORRC: Ouranos Reconstruction of RDPS and CaPA 
## Overview
Ouranos Reconstruction of RDPS and CaPA (ORRC) version 1.0 is a dataset designed to approximate the Canadian Surface Reanalysis (CaSR) and provide continuity beyond the CaSR period. It is produced from the Regional Deterministic Prediction System (RDPS), with precipitation fields adjusted by blending RDPS with the Canadian Precipitation Analysis (CaPA).

ORRC v1.0 provides hourly surface and near-surface meteorological fields at 0.09° (~10 km) resolution on a rotated regular latitude–longitude grid covering North America, Central America, and nearly all of Greenland, from 2015 to present. It supports near-real-time climate monitoring and is intended for operational climate services, event monitoring, and the routine update of climate indicators without waiting for future CaSR releases.

More details about ORRC v1.0 are available in the [documentation](/exec/abese/ORRC/documentation/ORRC_methodology.pdf). 

This repository contains a minimal example of the ORRC workflow using one week of test data. It includes the core processing steps extracted from the broader production pipeline:

1. Download RDPS and CaPA test data from the Ouranos THREDDS Data Server.
2. Preprocess RDPS and CaPA data, including consistency and completeness checks of the raw files and interpolation onto the CaSR v3.2 grid.
3. Create hourly ORRC data by combining consecutive 6- to 12-hour forecast lead-time segments from each RDPS cycle for all variables and blending RDPS precipitation fields with CaPA precipitation. ORRC output is saved as one NetCDF file per day.
4. Convert ORRC variables and metadata to Climate and Forecast (CF) conventions and save each variable as a continuous `.zarr.zip` time series.

In production, the workflow is typically run weekly on newly available RDPS and CaPA data obtained from ECCC's high-performance computer GPSC-C. This repository provides a lightweight example for testing and demonstration purposes.

We also produce a bias-adjusted variant, ORRC-a, for a subset of daily variables in order to reduce discrepancies relative to CaSR v3.2. Details of the bias-adjustment method are described in Section 2.3.3 of the [documentation](documentation/ORRC_methodology.pdf). Because the adjustment requires data over the full adjustment period, it cannot be demonstrated with the test dataset provided here. The corresponding code and configuration file are nevertheless included in the repository for users interested in reproducing the adjustment for variables beyond those included in the ORRC-a dataset.

## Installation
The ORRC workflow runs in a Conda environment and requires several packages. Create the environment using `environment.yml`, then activate it:

```bash
conda env create -f environment.yml
conda activate orrc
```

## Configuration 
The example configuration file is located at:

```bash
code/data/config_orrc.yml
```

The paths in this configuration are set up for the test workflow and use relative paths so that the example can be run directly within the repository.

## How to run 
Run all commands from the repository root.

### 1. Download the test data
Download one week of RDPS and CaPA test data:

```bash
python code/download_testdata.py
```

This script saves 6- to 12-hour forecast lead times from each RDPS cycle at 00, 06, 12, and 18 UTC as separate NetCDF files in the RDPS directory. The RDPS files includes the variables specified in `code/data/config_orrc.yml`. It also saves CaPA 6-hour precipitation accumulations at 00, 06, 12, and 18 UTC as separate NetCDF files in the CaPA_coarse directory.

### 2. Preprocess RDPS and CaPA
Preprocess the downloaded RDPS and CaPA files:

```bash
python code/preprocess_rdps_capa.py
```

This step performs completeness checks, identifies missing files and variables, fills missing content with `NaN` where needed, and interpolates RDPS and CaPA onto the CaSR v3.2 grid using the target file provided in the repository.

### 3. Create ORRC
Create the ORRC dataset:

```bash
python code/preprocess_orrc.py
```

To generate ORRC, consecutive RDPS forecast segments are concatenated to form a continuous hourly time series. The resulting 24-hour reporting window spans from 13 UTC to 12 UTC of the following calendar day. To produce a CaSR-like precipitation field, hourly RDPS precipitation increments are scaled so that each 6-hour accumulation matches the corresponding CaPA total. A more detailed description of the precipitation blending procedure is available in Section 2.3.2 of the [documentation](documentation/ORRC_methodology.pdf).

### 4. Convert to CF-compliant output
Apply Climate and Forecast (CF) variable and metadata conventions using the Miranda library, then save each variable as a separate `.zarr.zip` time series in the staging directory specified in the configuration file:

```bash
python code/convert_orrc.py
```

### Extra: Bias adjustment
To run the bias adjustment, the user must first create a data catalog containing the paths to:
- the daily ORRC data to be adjusted, and
- the reference dataset (CaSR v3.2).

The `data_catalogs` entries in the configuration file must then be updated accordingly so that the unadjusted ORRC and reference datasets can be extracted. This step produces ORRC-a output and saves each adjusted variable as a separate `.zarr.zip` time series.

Because the bias-adjustment method requires the full adjustment period, it cannot be demonstrated using the one-week test dataset included in this repository.

## Performance
We evaluated ORRC against CaSR v3.2 over the overlapping 2019–2024 period for the full set of variables. Details of the evaluation are available in the [documentation](documentation/ORRC_methodology.pdf). The main conclusions are:

- Over the full spatial domain, agreement between ORRC and CaSR v3.2 is strongest for temperature, dew point, and pressure variables, with generally low normalized errors. Errors are larger for humidity, wind, and radiation, while precipitation is the most challenging variable, especially in dry regions and during seasons with low mean precipitation, where normalization inflates relative errors. Corresponding error maps are available in [documentation/figures](documentation/figures/nrmse_maps.zip).
- In the regional evaluation based on Bukovsky regions, ORRC v1.0 agrees well with CaSR v3.2 for most variables, particularly temperature, pressure, humidity, and radiation. The largest regional mismatches occur for precipitation and wind components, which show greater spread and error. Corresponding Taylor diagrams are available in [documentation/figures](documentation/figures/taylor_bukovsky.zip).
- The bias-adjustment step substantially improves the magnitude and the spatial patterns of bias relative to CaSR v3.2. ORRC-a v1.0 should therefore be prioritized for applications requiring comparison with a historical baseline. Corresponding bias maps are available in [documentation/figures](documentation/figures/bias_maps.zip).

## Data availability and download
ORRC v1.0 and ORRC-a v1.0 will be made available through the [Ouranos THREDDS Data Server](https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/birdhouse/ouranos/catalog.html).

