# Regional temperature variability from CMIP6
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

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
from functools import reduce

# locations
loc_out = "../Data/CMIP6/RegionTvar/"
loc_cmip6 = "/" ## CMIP6 raw data
loc_shp = "../Data/Shapefile/"
loc_pop = "../Data/GPW/"

# warnings
import warnings
warnings.filterwarnings("ignore",category=FutureWarning,message="'base' in .resample()")

# years
#y1_hist = 1850
#y2_hist = 2014
y1 = 1950
y2 = 2020
y1_hist = 1950
y2_hist = 2014
y1_ssp = 2015
y2_ssp = 2020
ssp = "ssp245"
## histnat has all the years within the histnat experiments
## but historical has to be spliced with an ssp scenario
## if we want to go past 2014


# Functions for shapefile averaging

def transform_from_latlon(lat, lon):
    # Written by Alex Gottlieb
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

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

# strip leapdays
def xr_strip_leapdays(x):

    # This function removes leap days (February 29) from a timeseries x
    # The timeseries x is assumed to have a functional "time" coordinate in the xarray style

    x_noleap = x.sel(time=~((x.time.dt.month == 2) & (x.time.dt.day == 29)))

    return(x_noleap)

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


print("getting desired models",flush=True)
# hist models
varname = "tas_day"
hist_models = np.array([x for x in sorted(os.listdir(loc_cmip6+"historical/"+varname+"/")) if (x.endswith(".nc"))])
hist_models_prefix = np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in hist_models])

# histnat models
histnat_models = np.array([x for x in sorted(os.listdir(loc_cmip6+"historical-nat/"+varname+"/")) if (x.endswith(".nc"))])
histnat_models_prefix = np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in histnat_models])

# ssp models
ssp_models = np.array([x for x in sorted(os.listdir(loc_cmip6+ssp+"/"+varname+"/")) if (x.endswith(".nc"))])
ssp_models_prefix = np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in ssp_models])

# intersection of all three
models = reduce(np.intersect1d,(hist_models_prefix,histnat_models_prefix,ssp_models_prefix))
#models = np.array([x for x in models if "FGOALS" not in x])

# limit models we already did
model_limit = []
models = models[[x.split("_")[0] not in model_limit for x in models]]
models = models[[x.split("_")[0] not in ["FGOALS-g3"] for x in models]] # nuke FGOALS becuase not all years available in tas_day hist-nat
print(models)

# Read subnational region shapefile
print("reading shapefile",flush=True)
shp = gp.read_file(loc_shp+"gadm36_1.shp")
id_shp = shp.GID_1.values
idnums = {i: k for i, k in enumerate(shp.GID_1)}
#isonums_rev = {k: i for i, k in enumerate(shp.ISO3)}
shapes = [(shape, n) for n, shape in enumerate(shp.geometry)]


# new grid
res = 1
lon_new = np.arange(1,359.0+res,res)
lat_new = np.arange(-89.0,89.0+res,res)

## read in population and regrid
print("reading population",flush=True)
population_2000 = (xr.open_dataset(loc_pop+"gpw_v4_population_count_1degree_lonflip.nc").data_vars["population"])[0,::-1,:]
lat_pop = population_2000.coords["latitude"]
lon_pop = population_2000.coords["longitude"]
pop = xr.DataArray(population_2000.values,
                  coords=[lat_pop,lon_pop],
                  dims=["lat","lon"])
pop_regrid = pop.interp(lat=lat_new,lon=lon_new)
pop_flip = flip_lon_ll(pop_regrid)


# loop through models
for m in models:
    print(m,flush=True)
    mname = m.split("_")[0]
    mreal = m.split("_")[1]

    model_hist = mname+"_historical_"+mreal
    model_ssp = mname+"_"+ssp+"_"+mreal
    model_histn = mname+"_hist-nat_"+mreal

    # read tasmax
    print("reading hist data",flush=True)
    ta_ds_hist = xr.open_mfdataset(loc_cmip6+"historical/"+varname+"/"+varname+"_"+model_hist+"*.nc",
                                concat_dim="time")
    tm_hist = ta_ds_hist.coords["time"].load()
    tm_hist_ind = (tm_hist.dt.year>=y1_hist)&(tm_hist.dt.year<=y2_hist)
    ta_hist = xr_strip_leapdays(ta_ds_hist.tas[tm_hist_ind,:,:].load())
    if ta_hist.max()>200:
        ta_hist = ta_hist-273.15

    ta_ds_ssp = xr.open_mfdataset(loc_cmip6+ssp+"/"+varname+"/"+varname+"_"+model_ssp+"*.nc",
                                concat_dim="time")
    tm_ssp = ta_ds_ssp.coords["time"].load()
    tm_ssp_ind = (tm_ssp.dt.year>=y1_ssp)&(tm_ssp.dt.year<=y2_ssp)
    ta_ssp = xr_strip_leapdays(ta_ds_ssp.tas[tm_ssp_ind,:,:].load())
    if ta_ssp.max()>200:
        ta_ssp = ta_ssp-273.15

    ta_hist_final = xr.concat([ta_hist,ta_ssp],dim="time")
    del([ta_hist,ta_ssp])

    print("reading historical-nat data",flush=True)
    ta_ds_histn = xr.open_mfdataset(loc_cmip6+"historical-nat/"+varname+"/"+varname+"_"+model_histn+"*.nc",
                                concat_dim="time")
    tm_histn = ta_ds_histn.coords["time"].load()
    tm_histn_ind = (tm_histn.dt.year>=y1)&(tm_histn.dt.year<=y2)
    ta_histnat_final = xr_strip_leapdays(ta_ds_histn.tas[tm_histn_ind,:,:].load())
    if ta_histnat_final.max()>200:
        ta_histnat_final = ta_histnat_final-273.15

    print("processing and interpolating",flush=True)
    # standardize calendar
    cal_noleap = xr.cftime_range(start=str(y1)+"-01-01",end=str(y2)+"-12-31",
                                freq="D",calendar="noleap")
    ta_hist_final.coords["time"] = cal_noleap
    ta_histnat_final.coords["time"] = cal_noleap

    # regrid
    if (("latitude" in ta_hist_final.coords)&("longitude" in ta_hist_final.coords)):
        ta_hist_final = ta_hist_final.rename({"latitude":"lat","longitude":"lon"})
    if (("latitude" in ta_histnat_final.coords)&("longitude" in ta_histnat_final.coords)):
        ta_histnat_final = ta_histnat_final.rename({"latitude":"lat","longitude":"lon"})
    ta_hist_interp = ta_hist_final.interp(lat=lat_new,lon=lon_new,method="linear")
    del(ta_hist_final)
    ta_histnat_interp = ta_histnat_final.interp(lat=lat_new,lon=lon_new,method="linear")
    del(ta_histnat_final)

    print("calculating variability",flush=True)

    # tvar
    def monthly_to_yearly_mean(x):

        # calculate annual mean from monthly data
        # after weighting for the difference in month length
        # x must be data-array with time coord
        # xarray must be installed

        # x_yr = x.resample(time="YS").mean(dim="time") is wrong
        # because it doesn't weight for the # of days in each month

        days_in_mon = x.time.dt.days_in_month
        wgts = days_in_mon.groupby("time.year")/days_in_mon.groupby("time.year").sum()
        ones = xr.where(x.isnull(),0.0,1.0)
        x_sum = (x*wgts).resample(time="YS").sum(dim="time")
        ones_out = (ones*wgts).resample(time="YS").sum(dim="time")
        return(x_sum/ones_out)

    tvar_mon_hist = ta_hist_interp.resample(time="MS").std(dim="time")
    tvar_mon_hist.coords["time"] = pd.date_range(start=str(y1)+"-01-01",end=str(y2)+"-12-31",freq="MS")
    tvar_hist = monthly_to_yearly_mean(tvar_mon_hist)
    #tvar_hist = tvar_mon_hist.resample(time="YS").mean(dim="time")
    if m==models[0]:
        print(tvar_mon_hist,flush=True)
        print(tvar_hist,flush=True)
    del([tvar_mon_hist,ta_hist_interp])
    tvar_mon_histnat = ta_histnat_interp.resample(time="MS").std(dim="time")
    tvar_mon_histnat.coords["time"] = pd.date_range(start=str(y1)+"-01-01",end=str(y2)+"-12-31",freq="MS")
    tvar_histnat = monthly_to_yearly_mean(tvar_mon_histnat)
    #tvar_histnat = tvar_mon_histnat.resample(time="YS").mean(dim="time")
    del([tvar_mon_histnat,ta_histnat_interp])

    print("calculating regional averages",flush=True)
    def calc_region_average(x):
        xflip = flip_lon_tll(x)
        xregion = xr_region_average(xflip,y1,y2,"YS",shapes,id_shp,idnums,True,pop_flip)
        return(xregion)

    tvar_hist_region = calc_region_average(tvar_hist)
    tvar_histnat_region = calc_region_average(tvar_histnat)
    print((tvar_hist_region.mean(dim="region") - tvar_histnat_region.mean(dim="region")),flush=True)

    print("combining into dataframe and writing out",flush=True)

    hist_ds = xr.Dataset({"tvar":(["region","time"],tvar_hist_region)},
                          coords={"region":(["region"],tvar_hist_region.coords["region"]),
                                  "time":(["time"],tvar_hist_region.coords["time"])})

    hist_ds.attrs["creation_date"] = str(datetime.datetime.now())
    hist_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
    hist_ds.attrs["variable_description"] = "Daily-scale temperature variability at adm1 region level"
    hist_ds.attrs["created_from"] = os.getcwd()+"/CMIP6_Region_Tvar.py"

    fname_out = loc_out+m+"_historical-"+ssp+"_tvar_region_annual_"+str(y1)+"-"+str(y2)+".nc"
    hist_ds.to_netcdf(fname_out,mode="w")
    print(fname_out,flush=True)



    histnat_ds = xr.Dataset({"tvar":(["region","time"],tvar_histnat_region)},
                          coords={"region":(["region"],tvar_histnat_region.coords["region"]),
                                  "time":(["time"],tvar_histnat_region.coords["time"])})

    histnat_ds.attrs["creation_date"] = str(datetime.datetime.now())
    histnat_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
    histnat_ds.attrs["variable_description"] = "Daily-scale temperature variability at adm1 region level"
    histnat_ds.attrs["created_from"] = os.getcwd()+"/CMIP6_Region_Tvar.py"

    fname_out = loc_out+m+"_historical-nat_tvar_region_annual_"+str(y1)+"-"+str(y2)+".nc"
    histnat_ds.to_netcdf(fname_out,mode="w")
    print(fname_out,flush=True)
