# Globally unequal effect of extreme heat on economic growth

This repository provides replication data and code for the paper "National attribution of historical climate damages," by Christopher W. Callahan and Justin S. Mankin, published in _Science Advances_ (10.1126/sciadv.add3726).

### Overview

The repository is organized into **Scripts/**, **Figures/**, and **Data/** folders.

- **Scripts/**: All code required to reproduce the findings of our work is included in this folder. Most of the code is provided in Jupyter notebooks, except for major scripts like *Regional\_HeatWave\_Damages.py*, which requires batch processing on a high-performance computing cluster.

- **Figures/**: The final figures, both in the main text and the supplement, are included in this folder.

- **Data/**: This folder includes intermediate and processed summary data that enable replication of all the figures and numbers cited in the text. We do not provide some raw data due to large file sizes, specifically the ERA5 reanalysis data and CMIP6 climate model data. The ERA5 data can be downloaded from https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5, the CMIP6 data can be downloaded from the Earth System Grid Federation at https://esgf-node.llnl.gov/search/cmip6/, GADM shapefile data are publicly available at https://gadm.org/download_world.html and Palmer Drought Severity Index data are publicly available at https://crudata.uea.ac.uk/cru/data/drought/. The original economic data at the subnational level can be found in the repository for Kotz et al., Nature Climate Change, 2021, here: https://zenodo.org/record/4323163#.YxmZpOxByDU. Finally, the full dataset of global subnational GDP data was too big to provide here with each individual uncertainty sample (n=1000), but we provide the average and standard deviation of the samples for other users. If you'd like some or all of this raw data, send me an email at Christopher (dot) W (dot) Callahan (dot) GR (at) dartmouth.edu.
