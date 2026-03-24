# Ouranos Reconstruction – RDPS-CaPA (ORRC) v1.0 / Reconstruction Ouranos – RDPS‑CAPA (RORC) v1.0
## Overview
Ouranos Reconstruction – RDPS-CaPA (ORRC) version 1.0 is a dataset designed to approximate the Canadian Surface Reanalysis (CaSR) and provide continuity beyond the CaSR period. It is produced from the Regional Deterministic Prediction System (RDPS), with precipitation fields adjusted by blending RDPS with the Canadian Precipitation Analysis (CaPA).

ORRC v1.0 provides hourly surface and near-surface meteorological fields at 0.09° (~10 km) resolution on a rotated regular latitude–longitude grid covering North America, Central America, and nearly all of Greenland, from 2015 to present. It supports near-real-time climate monitoring and is intended for operational climate services, event monitoring, and the routine update of climate indicators without waiting for future CaSR releases.

More details about ORRC v1.0 are available in the [documentation](ORRC/documentation/ORRC_v10.pdf). 

This repository contains a minimal example of the ORRC workflow using one week of test data. It includes the core processing steps extracted from the broader production pipeline:

1. Download RDPS, CaPA and CaSR v3.2 test data from the Ouranos THREDDS Data Server.
2. Preprocess RDPS and CaPA data, including consistency and completeness checks of the raw files and interpolation onto the CaSR v3.2 grid.
3. Create hourly ORRC data by combining consecutive 6- to 12-hour forecast lead-time segments from each RDPS cycle for all variables and blending RDPS precipitation fields with CaPA precipitation. ORRC output is saved as one NetCDF file per day.
4. Convert ORRC variables and metadata to Climate and Forecast (CF) conventions, produce daily aggregates, and save each variable as a continuous `.zarr.zip` time series.
5. Produce a bias-adjusted variant, ORRC-a, to reduce discrepancies relative to CaSR v3.2.

In production, the workflow is typically run weekly on newly available RDPS and CaPA data obtained from ECCC's high-performance computer GPSC-C. This repository provides a lightweight example for testing and demonstration purposes.

## Installation
The ORRC workflow runs in a Conda environment and requires several packages. Create the environment using `environment.yml`, then activate it:

```bash
conda env create -f environment.yml
conda activate orrc
```

## Configuration 
The main ORRC workflow configuration file is located at:

```bash
code/data/config_orrc.yml
```

The bias-adjustment step uses a separate configuration file located at:

```bash
code/data/config_biasadj.yml
```

The paths in these configuration files are set up for the test workflow and use relative paths so that the example can be run directly within the repository.

## How to run 
Run all commands from the repository root.

### 0. Conversion from FST to NetCDF
RDPS and CaPA files are retrieved from GPSC-C in FST file format via a GPSC-C collaborator account. In the production workflow, these files are first converted to NetCDF using the [fst2nc Python utilities](https://github.com/neishm/fstd2nc). This test workflow downloads the test data hosted on the Ouranos THREDDS Data Server directly in NetCDF format so the conversion step is not included here.  

### 1. Download the test data
Download one week of RDPS and CaPA test data, as well as CaSR v3.2 target data for regridding:

```bash
python code/download_testdata.py
```

This script saves 6- to 12-hour forecast lead times from each RDPS cycle at 00, 06, 12, and 18 UTC as separate NetCDF files in the RDPS directory. The RDPS files includes the variables specified in `code/data/config_orrc.yml`. It also saves CaPA 6-hour precipitation accumulations at 00, 06, 12, and 18 UTC as separate NetCDF files in the CaPA_coarse directory, along with one day of CaSR v3.2 data used to regrid RDPS and CaPA data in the CaSR_v32 directory.

### 2. Preprocess RDPS and CaPA
Preprocess the downloaded RDPS and CaPA files:

```bash
python code/preprocess_rdps_capa.py
```

This step performs completeness checks, identifies missing files and variables, fills missing content with `NaN` where needed, and interpolates RDPS and CaPA onto the CaSR v3.2 grid.

### 3. Create ORRC
Create the hourly ORRC dataset:

```bash
python code/create_orrc.py
```

To generate ORRC, consecutive RDPS forecast segments are concatenated to form a continuous hourly time series. The resulting 24-hour reporting window spans from 13 UTC to 12 UTC of the following calendar day. To produce a CaSR-like precipitation field, hourly RDPS precipitation increments are scaled so that each 6-hour accumulation matches the corresponding CaPA total. A more detailed description of the precipitation blending procedure is available in Section 2.3.2 of the [documentation](documentation/ORRC_v10.pdf).

### 4. Convert to CF-compliant output and compute daily aggregates
Apply Climate and Forecast (CF) variable and metadata conventions using the Miranda library, compute daily aggregates as well as daily minimum and maximum for the specified variables, and save each variable as a separate `.zarr.zip` time series in hourly and daily formats in the staging directory defined in the configuration file:

```bash
python code/convert_orrc.py
```

### 5. Bias adjustment
Apply bias adjustment to align ORRC v1.0 with CaSR v3.2:

```bash
python code/adjust_bias.py
```

This step produces the bias-adjusted variant of ORRC (ORRC-a) data for a subset of daily variables, according to the specifications in `code/data/config_biasadj.yml`. The adjustment is trained over the reference period defined in that configuration file, using the ORRC dataset available from the [Ouranos THREDDS Data Server](https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/birdhouse/ouranos/catalog.html) as the product to be adjusted and CaSR v3.2 as the reference dataset. Each bias-adjusted variable is saved as a separate `.zarr.zip` time series. Details of the bias-adjustment method are described in Section 2.3.3 of the [documentation](documentation/ORRC_v10.pdf). 

The bias-adjustment workflow is designed to support two application modes:

- `apply_on: url` applies the bias adjustment to the full ORRC dataset retrieved from the THREDDS URL specified in the bias-adjustment configuration.
- `apply_on: staging` applies the bias adjustment only to matching ORRC data available in the local staging directory, which is useful for testing the workflow on newly created data.

Users who want to bias-adjust variables beyond those included in the ORRC-a v1.0 dataset should add a new entry for each variable in the bias-adjustment configuration file and set the appropriate `apply_on` mode depending on whether they want to process the full dataset from THREDDS or only the locally generated staging data.

## Performance
We evaluated ORRC against CaSR v3.2 over the overlapping 2019–2024 period for the full set of variables. Details of the evaluation are available in the [documentation](documentation/ORRC_v10.pdf). The main conclusions are:

- Over the full spatial domain, agreement between ORRC and CaSR v3.2 is strongest for temperature, dew point, and pressure variables, with generally low normalized errors. Errors are larger for humidity, wind, and radiation, while precipitation is the most challenging variable, especially in dry regions and during seasons with low mean precipitation, where normalization inflates relative errors. Corresponding error maps are available in [documentation/figures/nrmse](documentation/figures/nrmse/).
- In the regional evaluation based on Bukovsky regions, ORRC agrees well with CaSR v3.2 for most variables, particularly temperature, dew point, pressure, and radiation. The largest regional mismatches occur for relative humidity and precipitation, which show greater spread and error. Corresponding Taylor diagrams are available in [documentation/figures/taylor_diagrams](documentation/figures/taylor_diagrams).
- The bias-adjustment step substantially improves the magnitude and the spatial patterns of bias relative to CaSR v3.2. ORRC-a v1.0 should therefore be prioritized for applications requiring comparison with a historical baseline. Corresponding bias maps are available in [documentation/figures/bias](documentation/figures/bias).

## Data availability and download
ORRC v1.0 and ORRC-a v1.0 will be made available through the [Ouranos THREDDS Data Server](https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/birdhouse/ouranos/catalog.html).

