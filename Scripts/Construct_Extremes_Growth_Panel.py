# Panel data for extreme temperatures and economic growth
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

#### Mechanics
# Dependencies

import xarray as xr
import numpy as np
import sys
import os
import datetime
import pandas as pd
from rasterio import features
from affine import Affine
import geopandas as gp
import descartes

# Data locations

loc_shp = "../Data/Shapefile/"
loc_region_edd = "../Data/RegionMeans/EDD/"
loc_region_temp = "../Data/RegionMeans/Temperature/"
loc_region_seas = "../Data/RegionMeans/AnnualCycle/"
loc_region_pdsi = "../Data/RegionMeans/PDSI/"
loc_region_pop = "../Data/Population/"
loc_precip = "/" ## ERA5 total precip data -- https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
loc_pop = "../Data/GPW/"
loc_gdp_deflator = "../Data/GDP_Deflator/"
loc_panel = "../Data/Panel/"
loc_pwt = "../Data/PWT/"
loc_thresholds = "../Data/Thresholds/"
loc_nl = "../Data/Nightlights/"

# Year bounds

y1 = 1979
y2 = 2016
y1_edd = 1979
y2_edd = 2016

# Read subnational region shapefile

shp = gp.read_file(loc_shp+"gadm36_1.shp")
id_shp = shp.GID_1.values
idnums = {i: k for i, k in enumerate(shp.GID_1)}
#isonums_rev = {k: i for i, k in enumerate(shp.ISO3)}
shapes = [(shape, n) for n, shape in enumerate(shp.geometry)]


# Functions for shapefile averaging

def transform_from_latlon(lat, lon):
    # Written by Alex Gottlieb
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize_one(shape, latitude, longitude, fill=0, **kwargs):
    """Rasterize a shapefile geometry (polygon) onto the given
    xarray coordinates. This only works for 1d latitude and longitude
    arrays.
    Written by Alex Gottlieb and modified by Chris Callahan
    April 2020
    """
    transform = transform_from_latlon(latitude, longitude)
    out_shape = (len(latitude), len(longitude))
    raster = features.rasterize(shape, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    return xr.DataArray(raster, coords=[latitude,longitude], dims=["lat","lon"])

def rasterize(shapes, coords, fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xarray coordinates. This only works for 1d latitude and longitude
    arrays.
    """
    transform = transform_from_latlon(coords['lat'], coords['lon'])
    out_shape = (len(coords['lat']), len(coords['lon']))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    return xr.DataArray(raster, coords=coords, dims=('lat', 'lon'))


# Functions for flipping longitude

def flip_lon_tll(da):
    # flip 360 to 180 lon
    # for time-lat-lon xarray dataarray

    # get coords
    lat_da = da.coords["lat"]
    lon_da = da.coords["lon"]
    time_da = da.coords["time"]

    # flip lon
    lon_180 = (lon_da.values + 180) % 360 - 180

    # new data array
    da_180 = xr.DataArray(da.values,
                          coords=[time_da,lat_da.values,lon_180],
                          dims=["time","lat","lon"])

    # flip dataarray so it goes from -180 to 180
    # (instead of 0-180, -180-0)
    lon_min_neg = np.amin(lon_180[lon_180<0])
    lon_max_neg = np.amax(lon_180[lon_180<0])
    lon_min_pos = np.amin(lon_180[lon_180>=0])
    lon_max_pos = np.amax(lon_180[lon_180>=0])
    da_180_flip = xr.concat([da_180.loc[:,:,lon_min_neg:lon_max_neg],
                             da_180.loc[:,:,lon_min_pos:lon_max_pos]],
                            dim="lon")
    return(da_180_flip)

def flip_lon_ll(da):
    # flip 360 to 180 lon
    # for lat-lon xarray dataarray

    # get coords
    lat_da = da.coords["lat"]
    lon_da = da.coords["lon"]

    # flip lon
    lon_180 = (lon_da.values + 180) % 360 - 180

    # new data array
    da_180 = xr.DataArray(da.values,
                          coords=[lat_da,lon_180],
                          dims=["lat","lon"])

    # flip dataarray so it goes from -180 to 180
    # (instead of 0-180, -180-0)
    lon_min_neg = np.amin(lon_180[lon_180<0])
    lon_max_neg = np.amax(lon_180[lon_180<0])
    lon_min_pos = np.amin(lon_180[lon_180>=0])
    lon_max_pos = np.amax(lon_180[lon_180>=0])
    da_180_flip = xr.concat([da_180.loc[:,lon_min_neg:lon_max_neg],
                             da_180.loc[:,lon_min_pos:lon_max_pos]],
                            dim="lon")
    return(da_180_flip)


# Function for region averaging

def xr_region_average(data,y1,y2,freq,shapes,id_list,idnums_dict,weight=False,weightdata=None):

    time = pd.date_range(start=str(y1)+"-01-01",end=str(y2)+"-12-31",freq=freq)
    if freq=="D":
        time = time[(time.month!=2)|(time.day!=29)]
    region_array = xr.DataArray(np.full((len(id_list),len(time)),np.nan),
                                 coords=[id_list,time],
                                 dims=["region","time"])

    idraster = rasterize(shapes,data.drop("time").coords)
    data.coords["region"] = idraster
    if weight:
        weightdata.coords["region"] = idraster
        regionmean = ((data * weightdata).groupby("region").sum())/(weightdata.groupby("region").sum())
    else:
        regionmean = data.groupby("region").mean()

    idcoord = np.array([idnums_dict[n] for n in regionmean.coords["region"].values])
    region_array.loc[idcoord,:] = regionmean.transpose("region","time").values
    return(region_array)



#### Analysis

# Read in Kotz et al data

kotz_panel = pd.read_stata(loc_panel+"T_econ.dta")
econ_panel = kotz_panel.loc[:,["iso","T5_varm",
                        "wrld1id_1","lgdp_pc_usd","dlgdp_pc_usd","yearn",
                        "rich_dummy_iso","rich_dummy_ID",
                        "centroid_lat","centroid_lon"]]
econ_panel = econ_panel.rename(columns={"T5_varm":"var","yearn":"time"})
econ_panel["region"] = econ_panel["iso"]+"."+(econ_panel["wrld1id_1"].astype(int)).astype(str)+"_1"

## there are some differences between the kotz et al variability calculation
## and our variability calculation
## so we'll use theirs

# Population
pop_halfdegree = (xr.open_dataset(loc_pop+"gpw_v4_une_atotpopbt_cntm_30_min_lonflip.nc").data_vars["population"])[0,::-1,:]
pop_flip_halfdegree = flip_lon_ll(pop_halfdegree.rename({"latitude":"lat","longitude":"lon"}))
lat_min_halfdegree = np.amin(pop_flip_halfdegree.lat.values)
lat_max_halfdegree = np.amax(pop_flip_halfdegree.lat.values)

## read GDP deflator and adjust for inflation
gdp_deflator = pd.read_csv(loc_gdp_deflator+"API_NY.GDP.DEFL.ZS_DS2_en_excel_v2_4353530.csv")
usa_deflator = gdp_deflator.loc[gdp_deflator["Country Code"]=="USA",:].iloc[:,4:]
usa_deflator_xr = xr.DataArray(usa_deflator.values[0],coords=[np.arange(1960,2021+1,1)],dims=["time"])

target_year = 2010
deflator_ratio = usa_deflator_xr.loc[target_year]/usa_deflator_xr
del([gdp_deflator,usa_deflator,usa_deflator_xr])

# Read in temperature and seasonality data
print("getting temperature and annual cycle data",flush=True)
region_temp_ds = xr.open_dataset(loc_region_temp+"ERA5_temperature_adm1_region_"+str(y1)+"-"+str(y2)+".nc")
region_temp_ds.coords["time"] = region_temp_ds.time.dt.year.values
t = region_temp_ds.t_annual
t.name = "t"
# create panel
panel = t.to_dataframe().reset_index()
panel = pd.merge(panel,econ_panel,on=["region","time"],how="left")
panel["iso"] = [x[0:3] for x in panel.region.values]

# add gdp_deflator and calculate 2010 dollars
deflator_ratio.name = "gdp_deflator_2010"
print(deflator_ratio.to_dataframe().reset_index())
print(panel)
panel = pd.merge(panel,deflator_ratio.to_dataframe().reset_index(),on="time",how="left")
panel["gdppc_2010"] = np.exp(panel.lgdp_pc_usd)*panel.gdp_deflator_2010
panel["lgdppc_2010"] = np.log(panel.gdppc_2010)

# now other vars
tx_var = region_temp_ds.tx_variability
tx_var.name = "tx_var"
panel = pd.merge(panel,tx_var.to_dataframe().reset_index(),
                on=["region","time"],how="left")

tm = region_temp_ds.t_longterm
tm.name = "tmean"
panel = pd.merge(panel,tm.to_dataframe().reset_index(),
                on=["region","time"],how="left")

t_summer = region_temp_ds.t_summer
t_summer.name = "t_summer"
panel = pd.merge(panel,t_summer.to_dataframe().reset_index(),
                on=["region","time"],how="left")

t_winter = region_temp_ds.t_winter
t_winter.name = "t_winter"
panel = pd.merge(panel,t_winter.to_dataframe().reset_index(),
                on=["region","time"],how="left")

txx = region_temp_ds.txx
txx.name = "txx"
panel = pd.merge(panel,txx.to_dataframe().reset_index(),
                on=["region","time"],how="left")

tmonx = region_temp_ds.tmonx
tmonx.name = "tmonx"
panel = pd.merge(panel,tmonx.to_dataframe().reset_index(),
                on=["region","time"],how="left")

day_periods = region_temp_ds.day_period.values
for d in day_periods:
    txd = region_temp_ds.txd_running.loc[d,:,:]
    txd.name = "tx"+str(d)+"d"
    txd_df = txd.to_dataframe().reset_index().drop(columns=["day_period"])
    panel = pd.merge(panel,txd_df,on=["region","time"],how="left")
print(panel)

# get annual cycle and add
ann_cycle_in = xr.open_dataset(loc_region_seas+"ERA5_temperature_seasonality_adm1_region_"+str(y1)+"-"+str(y2)+".nc")
ann_cycle_annual = ann_cycle_in.annual_cycle_annual
ann_cycle_longterm = ann_cycle_in.annual_cycle_longterm
ann_cycle_annual.name = "seas_ann"
ann_cycle_annual.coords["time"] = ann_cycle_annual.time.dt.year.values
ann_cycle_longterm.name = "seas"
ann_cycle_longterm.coords["time"] = ann_cycle_longterm.time.dt.year.values

ac_annual_df = ann_cycle_annual.to_dataframe().reset_index()
ac_longterm_df = ann_cycle_longterm.to_dataframe().reset_index()

panel = pd.merge(panel,ac_annual_df,on=["region","time"],how="left")
panel = pd.merge(panel,ac_longterm_df,on=["region","time"],how="left")


print("getting EDD data",flush=True)
# Loop through thresholds, get EDD
threshold_type = "Month" #"Month" #"DayofYear"
#thresholds = np.arange(85,99+1,1)
thresholds = [95]
threshold_for_full_panel = 95
# if t==threshold for full panel -- edd_for_full_panel = edd_annual_region
# edd98, edd98_notcons, edd98_{seas}
for tt in np.arange(0,len(thresholds),1):
    t = thresholds[tt]
    print(t,flush=True)
    edd_in = xr.open_dataset(loc_region_edd+"ERA5_extreme_degree_days_"+threshold_type.lower()+"_percentile"+str(t)+"_adm1_region_"+str(y1_edd)+"-"+str(y2_edd)+".nc")
    edd_in.coords["time"] = edd_in.time.dt.year.values
    edd = edd_in.edd_consecutive
    edd_noncons = edd_in.edd_non_consecutive
    #edd_seasonal = edd_in.edd_seasonal
    #window_codes = edd_in.window_codes
    edd.name = "edd"+str(t)
    #edd.coords["time"] = edd.time.dt.year.values
    edd_noncons.name = "edd"+str(t)+"_noncons"
    edd_df = edd.to_dataframe().reset_index()
    edd_noncons_df = edd_noncons.to_dataframe().reset_index()
    panel = pd.merge(panel,edd_df,on=["region","time"],how="left")
    panel = pd.merge(panel,edd_noncons_df,on=["region","time"],how="left")
    
    # overall summer and winter EDDs
    if "edd_summer" in edd_in.data_vars:
        edd_summer = edd_in.edd_summer
        edd_winter = edd_in.edd_winter
        edd_summer.name = "edd"+str(t)+"_summer"
        edd_winter.name = "edd"+str(t)+"_winter"
        edd_summer_df = edd_summer.to_dataframe().reset_index()
        edd_winter_df = edd_winter.to_dataframe().reset_index()
        panel = pd.merge(panel,edd_summer_df,on=["region","time"],how="left")
        panel = pd.merge(panel,edd_winter_df,on=["region","time"],how="left")


# precip
print("loading and calculating accumulated precip",flush=True)
precip_in = xr.open_mfdataset(loc_precip+"total_precip_halfdegree_*.nc",concat_dim="time").tp.load()
precip_flip = flip_lon_tll(precip_in[:,::-1,:].loc[str(y1)+"-01-01":str(y2)+"-12-31",:,:])
p = xr_region_average(precip_flip,y1,y2,"YS",
                    shapes,id_shp,idnums,True,pop_flip_halfdegree)
del([precip_flip,precip_in])
p.name = "p"
p.coords["time"] = p.time.dt.year.values
p_df = p.to_dataframe().reset_index()
panel = pd.merge(panel,p_df,on=["region","time"],how="left")

print("loading and merging PDSI",flush=True)
# PDSI
pdsi = xr.open_dataset(loc_region_pdsi+"CRU_summer_PDSI_adm1_region_"+str(y1)+"-"+str(y2)+".nc").pdsi
pdsi.name = "pdsi"
pdsi_df = pdsi.to_dataframe().reset_index()
panel = pd.merge(panel,pdsi_df,on=["region","time"],how="left")


# loop through panel and calculate growth
print("calculating growth",flush=True)
panel["growth"] = np.full(len(panel.region.values),np.nan)
for i in panel.index:
    if np.mod(i,5000)==0:
        print(i,flush=True)
    full_id = panel.loc[panel.index==i,"region"].values[0]
    year = panel.loc[panel.index==i,"time"].values[0]
    #gdp = np.exp(panel.loc[panel.index==i,"lgdp_pc_usd"].values[0])
    gdp = panel.loc[panel.index==i,"gdppc_2010"].values[0]
    if i>0:
        full_id_0 = panel.loc[panel.index==i-1,"region"].values[0]
        year_0 = panel.loc[panel.index==i-1,"time"].values[0]
        #gdp_0 = np.exp(panel.loc[panel.index==i-1,"lgdp_pc_usd"].values[0])
        gdp_0 = panel.loc[panel.index==i-1,"gdppc_2010"].values[0]
        if (full_id_0==full_id)&(year_0==(year-1)):
            panel.loc[panel.index==i,"growth"] = (gdp-gdp_0)/gdp_0


print("adding population and nightlights data",flush=True)
## population from GPW
y1_pop_region = 1991
y2_pop_region = 2015
pop_region_df = pd.read_csv(loc_region_pop+"GPW_population_interpolated_region_"+str(y1_pop_region)+"-"+str(y2_pop_region)+".csv",
                            index_col=0)
pop_region_df = pop_region_df.rename(columns={"id":"region"})
panel = panel.merge(pop_region_df,on=["region","time"],how="left")

## add nightlights
y1_nl = 1991
y2_nl = 2015
nightlights = pd.read_csv(loc_nl+"DSMP-OLS_stable_lights_region_"+str(y1_nl)+"-"+str(y2_nl)+".csv",
                            index_col=0)
nl = nightlights.rename(columns={"id":"region"}).loc[:,["region","time","luminosity"]]
panel = panel.merge(nl,how="left",on=["region","time"])


## add country-level info from the Penn World Tables
print("adding country-level values from the penn world tables",flush=True)
pwt_in = pd.read_csv(loc_pwt+"pwt10-0.csv",engine="python")
pwt_iso = pwt_in.countrycode.values
pwt_yr = pwt_in.year.values
pwt_in["population"] = pwt_in["pop"]*1e6 # originally in millions
pwt_in["gpc"] = (pwt_in["rgdpna"]*1e6)/pwt_in["population"]
pwt_in["gdp"] = pwt_in["rgdpna"]*1e6

# loop
panel["iso_gpc"] = np.full(len(panel.region.values),np.nan)
panel["iso_gr"] = np.full(len(panel.region.values),np.nan)
panel["iso_gdp"] = np.full(len(panel.region.values),np.nan)
panel["iso_pop"] = np.full(len(panel.region.values),np.nan)
for i in panel.index.values:
    if np.mod(i,5000)==0:
        print(i,flush=True)
    reg = str(panel.loc[panel.index==i,"region"].values[0])
    iso = str(reg[0:3])
    year = int(panel.loc[panel.index==i,"time"].values[0])

    if (iso in list(pwt_iso)) & (year in list(pwt_yr)):
        gpc_i = pwt_in.loc[(pwt_iso==iso)&(pwt_yr==year),"gpc"].values[0]
        gdp_i = pwt_in.loc[(pwt_iso==iso)&(pwt_yr==year),"gdp"].values[0]
        pop_i = pwt_in.loc[(pwt_iso==iso)&(pwt_yr==year),"population"].values[0]
        panel.loc[panel.index==i,"iso_gpc"] = gpc_i
        panel.loc[panel.index==i,"iso_gdp"] = gdp_i
        panel.loc[panel.index==i,"iso_pop"] = pop_i

    # growth
    if (iso in list(pwt_iso)) & (year in list(pwt_yr)) & (year > np.amin(pwt_yr)):
        gpc_1 = pwt_in.loc[(pwt_iso==iso)&(pwt_yr==year),"gpc"].values[0]
        gpc_0 = pwt_in.loc[(pwt_iso==iso)&(pwt_yr==year-1),"gpc"].values[0]
        panel.loc[panel.index==i,"iso_gr"] = (gpc_1 - gpc_0)/gpc_0

## write out!

fname = loc_panel+"extremes_growth_panel_"+threshold_type.lower()+"edd_"+str(y1)+"-"+str(y2)+".csv"
panel.to_csv(fname)
print(fname,flush=True)
