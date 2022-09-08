# Timing of hottest 5-day period at the grid cell level
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
loc_temp = "/" ## ERA5 temperature data -- https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
loc_pop = "../Data/GPW/"
loc_tmean = "/" ## UDel temperature data -- http://climate.geog.udel.edu/~climate/html_pages/download.html
loc_out = "../Data/Tx5d/"

# warnings

import warnings
warnings.filterwarnings("ignore",category=FutureWarning,message="'base' in .resample()")


# Year bounds

y1 = 1979
y2 = 2016
y1_tmean = 1979
y2_tmean = 2016
y1_clm = 1979
y2_clm = 2016
y1_edd = 1979
y2_edd = 2016

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

## leap days function

def xr_strip_leapdays(x):

    # This function removes leap days (February 29) from a timeseries x
    # The timeseries x is assumed to have a functional "time" coordinate in the xarray style

    x_noleap = x.sel(time=~((x.time.dt.month == 2) & (x.time.dt.day == 29)))

    return(x_noleap)




#### Analysis

# Read in temp
print("reading temperature data",flush=True)

tmax = xr.open_mfdataset(loc_temp+"tmax_daily_halfdegree_*.nc").tmax - 273.15

# limit to years
tmax = tmax.loc[str(y1)+"-01-01":str(y2)+"-12-31",:,:].load()

# tx5d
day_period = 5
print("calculating rolling mean",flush=True)
tmax_running = tmax.rolling(time=day_period,min_periods=day_period,center=True).mean()
tmax_running_shift = tmax_running.shift(time=182) # july 1
del(tmax)

print("calculating day of year for tx5d")
def max_doy(x):
    return(np.argmax(x)+1)

lat = tmax_running.lat.values
lon = tmax_running.lon.values
year = np.arange(y1,y2+1,1)

tx5d_doy = xr.DataArray(np.full((len(year),len(lat),len(lon)),np.nan),
                        coords=[year,lat,lon],dims=["time","lat","lon"])
tx5d_doy2 = xr.DataArray(np.full((len(year),len(lat),len(lon)),np.nan),
                        coords=[year,lat,lon],dims=["time","lat","lon"])

for j in np.arange(0,len(lat),1):
    for k in np.arange(0,len(lon),1):
        print(str(lat[j])+", "+str(lon[k]),flush=True)
        tmax_running_ind = tmax_running[:,j,k]
        tx5d_doy[:,j,k] = xr.apply_ufunc(max_doy,tmax_running[:,j,k].groupby("time.year"),
                                        input_core_dims=[["time"]],output_core_dims=[[]]).values
        tx5d_doy2[:,j,k] = xr.apply_ufunc(max_doy,tmax_running_shift[:,j,k].groupby("time.year"),
                                        input_core_dims=[["time"]],output_core_dims=[[]]).values

tx5d_doy.name = "dayofyear"
tx5d_doy.attrs["creation_date"] = str(datetime.datetime.now())
tx5d_doy.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
tx5d_doy.attrs["variable_description"] = "Day of year that each hottest 5-day period takes place (centered)"
tx5d_doy.attrs["created_from"] = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/Extremes_Economics/Scripts_v2/Calculate_Tx5d_DayofYear.py"

fname_out = loc_out+"ERA5_Tx5d_dayofyear_"+str(y1)+"-"+str(y2)+".nc"
tx5d_doy.to_netcdf(path=fname_out,mode="w")
print(fname_out,flush=True)

# replace NH values of tx5d_doy2 with tx5d_doy
tx5d_doy2.loc[:,89.75:0.25,:] = tx5d_doy.loc[:,89.75:0.25,:]*1.0
tx5d_doy2.name = "dayofyear"
tx5d_doy2.attrs["creation_date"] = str(datetime.datetime.now())
tx5d_doy2.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
tx5d_doy2.attrs["variable_description"] = "Day of year that each hottest 5-day period takes place (centered), with the southern hemisphere shifted by half a year"
tx5d_doy2.attrs["created_from"] = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/Extremes_Economics/Scripts_v2/Calculate_Tx5d_DayofYear.py"

fname_out = loc_out+"ERA5_Tx5d_dayofyear_shshift_"+str(y1)+"-"+str(y2)+".nc"
tx5d_doy2.to_netcdf(path=fname_out,mode="w")
print(fname_out,flush=True)
