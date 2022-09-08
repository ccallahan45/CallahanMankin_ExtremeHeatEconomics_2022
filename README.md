# Globally unequal effect of extreme heat on economic growth

This repository provides replication data and code for the paper "National attribution of historical climate damages," by Christopher W. Callahan and Justin S. Mankin, published in _Science Advances_ (doi to come).

### Overview

The repository is organized into **Scripts/**, **Figures/**, and **Data/** folders.

- **Scripts/**: All code required to reproduce the findings of our work is included in this folder. Most of the code is provided in Jupyter notebooks, except for major scripts like *Regional\_HeatWave\_Damages.py*, which requires batch processing on a high-performance computing cluster.

- **Figures/**: The final figures, both in the main text and the supplement, are included in this folder.

- **Data/**: This folder includes intermediate and processed summary data that enable replication of all the figures and numbers cited in the text. We do not provide some raw data due to large file sizes, specifically the ERA5 reanalysis data and CMIP6 climate model data. The ERA5 data can be downloaded from https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5 and the CMIP6 data can be downloaded from the Earth System Grid Federation at https://esgf-node.llnl.gov/search/cmip6/. Finally, the original economic data at the subnational level can be found in the repository for Kotz et al., Nature Climate Change, 2021, here: https://zenodo.org/record/4323163#.YxmZpOxByDU

### Details of specific scripts
