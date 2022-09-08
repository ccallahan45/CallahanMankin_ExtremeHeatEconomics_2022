# Temperature and extreme degree days in subnational regions
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

## takes several hours to run
## depending on number of EDD thresholds used
## on an HPC system -- would not recommend running on a local machine


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
loc_temp = "/" ## ERA5 tx data -- https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
loc_pdsi = "../Data/PDSI/"
loc_pop = "../Data/GPW/"
loc_tmean = "/" ## UDel temperature data -- http://climate.geog.udel.edu/~climate/html_pages/download.html
loc_thresholds = "../Data/Thresholds/"
loc_edd_out = "../Data/RegionMeans/EDD/"
loc_t_out = "../Data/RegionMeans/Temperature/"
loc_pdsi_out = "../Data/RegionMeans/PDSI/"
loc_seas_out = "../Data/RegionMeans/AnnualCycle/"


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

# Read subnational region shapefile
print("reading shapefile and population data",flush=True)
shp = gp.read_file(loc_shp+"gadm36_1.shp")
id_shp = shp.GID_1.values
idnums = {i: k for i, k in enumerate(shp.GID_1)}
#isonums_rev = {k: i for i, k in enumerate(shp.ISO3)}
shapes = [(shape, n) for n, shape in enumerate(shp.geometry)]

# Population
pop_halfdegree = (xr.open_dataset(loc_pop+"gpw_v4_une_atotpopbt_cntm_30_min_lonflip.nc").data_vars["population"])[0,::-1,:]
pop_flip_halfdegree = flip_lon_ll(pop_halfdegree.rename({"latitude":"lat","longitude":"lon"}))
lat_min_halfdegree = np.amin(pop_flip_halfdegree.lat.values)
lat_max_halfdegree = np.amax(pop_flip_halfdegree.lat.values)


# Read in temp
print("reading temperature data",flush=True)

tmax = xr.open_mfdataset(loc_temp+"tmax_daily_halfdegree_*.nc").tmax - 273.15
tmin = xr.open_mfdataset(loc_temp+"tmin_daily_halfdegree_*.nc").tmin - 273.15
tmean = xr.open_dataset(loc_tmean+"air.mon.mean.v501.nc").air
tmean_daily = (tmax + tmin)/2.0
del(tmin)
if tmean.max()>200:
    t_mean = tmean - 273.15
else:
    t_mean = tmean*1.0
del(tmean)

# limit to years
tmax = tmax.loc[str(y1)+"-01-01":str(y2)+"-12-31",:,:].load()
t_mean = t_mean.loc[str(y1_tmean)+"-01-01":str(y2_tmean)+"-12-31",:,:]
tmean_daily = tmean_daily.loc[str(y1)+"-01-01":str(y2)+"-12-31",:,:].load()


# read PDSI for drought
y1_pdsi = 1901
y2_pdsi = 2019
pdsi_in = xr.open_dataset(loc_pdsi+"scPDSI.cru_ts4.04early1.1901.2019.cal_1901_19.bams.2020.GLOBAL."+str(y1_pdsi)+"."+str(y2_pdsi)+".nc").scpdsi
pdsi = pdsi_in.rename({"latitude":"lat","longitude":"lon"})

pdsi_jja = pdsi[(pdsi.time.dt.season=="JJA"),:,:].resample(time="YS").mean(dim="time")
pdsi_shift = pdsi.shift(time=1)
pdsi_djf = pdsi_shift[(pdsi_shift.time.dt.month<=3),:,:].resample(time="YS").mean(dim="time")
pdsi_final1 = xr.concat([pdsi_jja.loc[:,89.75:0.25,:],pdsi_djf.loc[:,-0.25:-89.75,:]],dim="lat")
pdsi_final = pdsi_final1[:,::-1,:]
pdsi_region = xr_region_average(pdsi_final,y1_pdsi,y2_pdsi,"YS",
                                shapes,id_shp,idnums,
                                True,pop_flip_halfdegree)
pdsi_region_out = pdsi_region.loc[:,str(y1)+"-01-01":str(y2)+"-12-31"]
pdsi_region_out.coords["time"] = pdsi_region_out.time.dt.year.values

# write out PDSI
pdsi_ds = xr.Dataset({"pdsi":(["region","time"],pdsi_region_out)},
                coords={"region":(["region"],pdsi_region_out.coords["region"]),
                        "time":(["time"],pdsi_region_out.coords["time"])})

pdsi_ds.attrs["creation_date"] = str(datetime.datetime.now())
pdsi_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
pdsi_ds.attrs["variable_description"] = "summer mean PDSI at adm1-level regions -- NH JJA, SH DJF"
pdsi_ds.attrs["created_from"] = os.getcwd()+"/Calculate_Regional_T_EDD.py"

fname_out = loc_pdsi_out+"CRU_summer_PDSI_adm1_region_"+str(y1)+"-"+str(y2)+".nc"
pdsi_ds.to_netcdf(fname_out,mode="w")
print(fname_out,flush=True)


# info for seasonal windows, etc
window = 3
mon_begin = np.arange(0,11+1,1)
month_codes = ["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA",\
                "JAS","ASO","SON","OND","NDJ"]
m1_summer_nh = 4 # apr
m2_summer_nh = 9 # sep

calc_averages = True # in case we want to skip this first bit
if calc_averages:
    # calculate annual mean temperature
    print("calculating annual mean temperature",flush=True)
    tmean_monthly = flip_lon_tll(t_mean[:,::-1,:])

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

    #tmean_ann = t_mean[:,::-1,:].resample(time="YS").mean(dim="time")
    tmean_ann = monthly_to_yearly_mean(t_mean[:,::-1,:])
    tmean_annual = flip_lon_tll(tmean_ann)
    del([tmean_ann])

    tmean_annual_region = xr_region_average(tmean_annual,y1_tmean,y2_tmean,"YS",
                                            shapes,id_shp,idnums,
                                            True,pop_flip_halfdegree)
    tmean_monthly_region = xr_region_average(tmean_monthly,y1_tmean,y2_tmean,"MS",
                                            shapes,id_shp,idnums,
                                            True,pop_flip_halfdegree)
    del(tmean_annual)
    tmean_region_longterm = tmean_annual_region.mean(dim="time")

    print("calculating summer/winter temperatures",flush=True)
    lat_vals = tmean_monthly.coords["lat"].values
    lat_cutoff = list(lat_vals).index(-0.25)
    months = tmean_monthly.time.dt.month
    summer_nh_ind = (months >= m1_summer_nh) & (months <= m2_summer_nh)
    summer_sh_ind = (months <= (m1_summer_nh-1)) | (months >= (m2_summer_nh+1))
    t_summer_nh = tmean_monthly[summer_nh_ind,lat_cutoff:,:].resample(time="YS").mean(dim="time")
    t_summer_sh = tmean_monthly[summer_sh_ind,:lat_cutoff,:].resample(time="YS").mean(dim="time")
    t_summer = xr.concat([t_summer_sh,t_summer_nh],dim="lat")
    t_summer_region = xr_region_average(t_summer,y1_tmean,y2_tmean,"YS",
                                      shapes,id_shp,idnums,True,pop_flip_halfdegree)

    winter_nh_ind = (months <= (m1_summer_nh-1)) | (months >= (m2_summer_nh+1))
    winter_sh_ind = (months >= m1_summer_nh) & (months <= m2_summer_nh)
    t_winter_nh = tmean_monthly[winter_nh_ind,lat_cutoff:,:].resample(time="YS").mean(dim="time")
    t_winter_sh = tmean_monthly[winter_sh_ind,:lat_cutoff,:].resample(time="YS").mean(dim="time")
    t_winter = xr.concat([t_winter_sh,t_winter_nh],dim="lat")
    t_winter_region = xr_region_average(t_winter,y1_tmean,y2_tmean,"YS",
                                      shapes,id_shp,idnums,True,pop_flip_halfdegree)

    print("calculating variability",flush=True)
    def calc_region_mean(x):
        xflip = flip_lon_tll(x[:,::-1,:])
        x_region = xr_region_average(xflip,y1,y2,"YS",
                                    shapes,id_shp,idnums,
                                    True,pop_flip_halfdegree)
        return(x_region)

    # average daily-scale standard deviation
    #tvar_mon = tmax.resample(time="MS").std(dim="time")
    #tvar = monthly_to_yearly_mean(tvar_mon)
    tvar_mon = tmean_daily.resample(time="MS").std(dim="time")
    tvar = monthly_to_yearly_mean(tvar_mon)
    tx_variability = calc_region_mean(tvar)

    print("calculating other metrics of extremes",flush=True)
    txx = tmax.resample(time="YS").max(dim="time")
    tmonx = tmax.resample(time="MS").mean(dim="time").resample(time="YS").max(dim="time")
    txx_region = calc_region_mean(txx)
    tmonx_region = calc_region_mean(tmonx)
    del([txx,tmonx])

    day_periods = [3,5,7,10,15]
    txd_running = xr.DataArray(np.full((len(day_periods),len(txx_region.region),len(txx_region.time)),np.nan),
                                    coords=[day_periods,txx_region.region,txx_region.time],
                                    dims=["day_period","region","time"])
    for day_period in day_periods:
        print(str(day_period)+"-day running mean",flush=True)
        tmax_running = tmax.rolling(time=day_period,min_periods=day_period,center=True).mean()
        txd_running.loc[day_period,:,:] = calc_region_mean(tmax_running.resample(time="YS").max(dim="time"))

    # create dataset and write out!
    ds = xr.Dataset({"t_annual":(["region","time"],tmean_annual_region),
                    "t_longterm":(["region","time"],tmean_region_longterm.expand_dims(time=tmean_annual_region.time).transpose("region","time")),
                    "t_summer":(["region","time"],t_summer_region),
                    "t_winter":(["region","time"],t_winter_region),
                    "tx_variability":(["region","time"],tx_variability),
                    "txd_running":(["day_period","region","time"],txd_running),
                    "txx":(["region","time"],txx_region),
                    "tmonx":(["region","time"],tmonx_region)},
                    coords={"region":(["region"],tmean_annual_region.coords["region"]),
                            "time":(["time"],tmean_annual_region.coords["time"]),
                            "day_period":(["day_period"],txd_running.coords["day_period"])})
                    #"t_seasonal":(["month_window","region","time"],t_seas_region),
                    #"window_codes":(["month_window"],month_codes)
                    #"month_window":(["month_window"],t_seas_region.coords["month_window"])
                    #"tx7d_multiple_hw":(["hw_num","region","time"],tx7d_hws)},
                    #"hw_num":(["hw_num"],tx7d_hws.coords["hw_num"])
                    #"tx7d_anom":(["region","time"],tx7d_anom_region),
                    #"tx7d":(["region","time"],tx7d_region),

    ds.attrs["creation_date"] = str(datetime.datetime.now())
    ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
    ds.attrs["variable_description"] = "Annual, seasonal, maximum/running, and long-term temperatures at adm1-level regions"
    ds.attrs["created_from"] = os.getcwd()+"/Calculate_Regional_T_EDD.py"

    fname_out = loc_t_out+"ERA5_temperature_adm1_region_"+str(y1)+"-"+str(y2)+".nc"
    ds.to_netcdf(fname_out,mode="w")
    print(fname_out,flush=True)


    # now calculate the annual and long-term seasonal cycle/annual cycle and write that out as well
    print("calculating seasonality",flush=True)
    monthly_max_temp = tmean_monthly_region.resample(time="YS").max(dim="time")
    monthly_min_temp = tmean_monthly_region.resample(time="YS").min(dim="time")
    seasonal_cycle_region = monthly_max_temp - monthly_min_temp
    seasonal_cycle_ann = seasonal_cycle_region.transpose("region","time")
    seasonal_cycle_mean = seasonal_cycle_region.mean(dim="time")
    seasonal_cycle_final = seasonal_cycle_mean.expand_dims(time=seasonal_cycle_region.time).transpose("region","time")

    # create dataset and write out!
    ds = xr.Dataset({"annual_cycle_annual":(["region","time"],seasonal_cycle_ann),
                    "annual_cycle_longterm":(["region","time"],seasonal_cycle_final)},
                    coords={"region":(["region"],seasonal_cycle_ann.coords["region"]),
                            "time":(["time"],seasonal_cycle_ann.coords["time"])})

    ds.attrs["creation_date"] = str(datetime.datetime.now())
    ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
    ds.attrs["variable_description"] = "Seasonality (i.e., annual cycle) at adm1-level regions"
    ds.attrs["created_from"] = os.getcwd()+"/Calculate_Regional_T_EDD.py"

    fname_out = loc_seas_out+"ERA5_temperature_seasonality_adm1_region_"+str(y1)+"-"+str(y2)+".nc"
    ds.to_netcdf(fname_out,mode="w")
    print(fname_out,flush=True)


# Loop through thresholds, calculate extreme degree days
print("calculating EDDs",flush=True)
thresholds = [95] #[90,95,98,99]
# np.arange(85,99+1,1)
threshold_type = "Month"

for tt in np.arange(0,len(thresholds),1):
    t = thresholds[tt]
    print(t,flush=True)

    fname_hd_threshold = "ERA5_HotDay_Threshold_"+threshold_type+str(t)+"_"+str(y1_clm)+"-"+str(y2_clm)+".nc"
    hd_threshold_1 = xr.DataArray(xr.open_dataset(loc_thresholds+fname_hd_threshold).data_vars["threshold"])
    #hd_threshold = xr.DataArray(hd_threshold_1.values,
    #                           coords=[pd.date_range(start="2000-01-01",end="2000-12-31"),hd_threshold_1.lat,hd_threshold_1.lon],
    #                           dims=["dayofyear","lat","lon"])

    if threshold_type=="DayofYear":
        hd_threshold_v2 = xr.DataArray(hd_threshold_1.values,
                                coords=[np.arange(1,366+1,1),hd_threshold_1.lat,hd_threshold_1.lon],
                                dims=["dayofyear","lat","lon"])
    elif threshold_type=="Month":
        hd_threshold_v2 = xr.DataArray(hd_threshold_1.values,
                                coords=[np.arange(1,12+1,1),hd_threshold_1.lat,hd_threshold_1.lon],
                                dims=["month","lat","lon"])
    elif threshold_type=="Season":
        hd_threshold_v2 = xr.DataArray(hd_threshold_1.values,
                                coords=[["DJF","JJA","MAM","SON"],hd_threshold_1.lat,hd_threshold_1.lon],
                                dims=["season","lat","lon"])
    elif threshold_type=="Overall":
        #hd_threshold_v2 = xr.DataArray(hd_threshold_1.expand_dims(month=np.arange(1,12+1,1)).values,
        #                        coords=[np.arange(1,12+1,1),hd_threshold_1.lat,hd_threshold_1.lon],
        #                        dims=["month","lat","lon"])
        hd_threshold_v2 = hd_threshold_1.expand_dims(month=np.arange(1,12+1,1))*1.0
    latmax = np.amax(hd_threshold_v2.lat.values)
    latmin = np.amin(hd_threshold_v2.lat.values)
    lonmax = np.amax(hd_threshold_v2.lon.values)
    lonmin = np.amin(hd_threshold_v2.lon.values)
    del(hd_threshold_1)

    #hd_threshold_1.coords["time"] = pd.date_range(start="2000-01-01",end="2000-12-31")
    # remove Feb 29
    #hd_threshold = hd_threshold_1[(hd_threshold_1.time.dt.month != 2) | (hd_threshold_1.time.dt.day != 29),:,:]
    print("processing threshold exceedance data",flush=True)
    tmax = tmax.loc[str(y1_edd)+"-01-01":str(y2_edd)+"-12-31",latmax:latmin,lonmin:lonmax]
    if threshold_type!="Overall":
        tmax_bool = (tmax.groupby("time."+threshold_type.lower()) > hd_threshold_v2).astype(float)
        print("calculating absolute difference",flush=True)
        tmax_diff = tmax.groupby("time."+threshold_type.lower()) - hd_threshold_v2
        print("done calculating absolute difference",flush=True)
    elif threshold_type=="Overall":
        tmax_bool = (tmax.groupby("time.month") > hd_threshold_v2).astype(float)
        print("calculating absolute difference",flush=True)
        tmax_diff = tmax.groupby("time.month") - hd_threshold_v2
        print("done calculating absolute difference",flush=True)

    del(hd_threshold_v2)

    print("shifting and calculating consecutive days",flush=True)
    # first do center-based rolling, then shift to left and right
    tmax_bool_run = tmax_bool.rolling(time=3,min_periods=3,center=True).sum()
    del(tmax_bool)
    tmax_bool_run_shiftleft = tmax_bool_run.shift(time=-1)
    tmax_bool_run_shiftright = tmax_bool_run.shift(time=1)

    tmax_bool_run_shift = tmax_bool_run.expand_dims("shift")
    del(tmax_bool_run)
    tmax_bool_run_shift = xr.concat([tmax_bool_run_shiftleft,tmax_bool_run_shift],dim="shift")
    del(tmax_bool_run_shiftleft)
    tmax_bool_run_shift = xr.concat([tmax_bool_run_shift,tmax_bool_run_shiftright],dim="shift")
    del(tmax_bool_run_shiftright)
    tmax_bool_run_shift.coords["shift"] = [-1,0,1]
    tmax_bool_run_three = (tmax_bool_run_shift.max(dim="shift") == 3).astype(int)
    del(tmax_bool_run_shift)

    print("calculating consecutive and non-consecutive EDDs",flush=True)
    # both consecutive and non-consecutive EDDs
    tmax_diff_consecutive = tmax_diff*tmax_bool_run_three
    tmax_extreme_consecutive = tmax_diff_consecutive.where(tmax_diff_consecutive>0,0)
    tmax_extreme_notcons = tmax_diff.where(tmax_diff>0,0.0)
    if threshold_type=="DayofYear":
        # get rid of leap days
        # since the threshold estimate for feb 29 is very unstable
        tmax_extreme_consecutive = xr_strip_leapdays(tmax_extreme_consecutive)
        tmax_extreme_notcons = xr_strip_leapdays(tmax_extreme_notcons)
    edd_annual = tmax_extreme_consecutive.resample(time="YS").sum(dim="time")
    edd_annual_notcons = tmax_extreme_notcons.resample(time="YS").sum(dim="time")

    print("calculating annual regional EDDs",flush=True)
    # average over regions
    edd_annual_flip = flip_lon_tll(edd_annual[:,::-1,:])
    edd_annual_notcons_flip = flip_lon_tll(edd_annual_notcons[:,::-1,:])
    del([edd_annual,edd_annual_notcons])
    edd_annual_region = xr_region_average(edd_annual_flip,y1_edd,y2_edd,"YS",
                                      shapes,id_shp,idnums,True,pop_flip_halfdegree)
    edd_annual_notcons_region = xr_region_average(edd_annual_notcons_flip,y1_edd,y2_edd,"YS",
                                                shapes,id_shp,idnums,True,pop_flip_halfdegree)


    edd_monthly = tmax_extreme_consecutive.resample(time="MS").sum(dim="time")
    
    # split up by lat for warm and cool seasons
    print("calculating summer and winter EDDs",flush=True)
    lat_vals = edd_monthly.coords["lat"].values
    lat_cutoff = list(lat_vals).index(-0.25)
    months = edd_monthly.time.dt.month
    summer_nh_ind = (months >= m1_summer_nh) & (months <= m2_summer_nh)
    summer_sh_ind = (months <= (m1_summer_nh-1)) | (months >= (m2_summer_nh+1))
    edd_summer_nh = edd_monthly[summer_nh_ind,:lat_cutoff,:].resample(time="YS").sum(dim="time")
    edd_summer_sh = edd_monthly[summer_sh_ind,lat_cutoff:,:].resample(time="YS").sum(dim="time")
    edd_summer = (xr.concat([edd_summer_nh,edd_summer_sh],dim="lat"))
    edd_summer_flip = flip_lon_tll(edd_summer[:,::-1,:])
    edd_summer_region = xr_region_average(edd_summer_flip,y1_edd,y2_edd,"YS",
                                      shapes,id_shp,idnums,True,pop_flip_halfdegree)

    winter_nh_ind = (months <= (m1_summer_nh-1)) | (months >= (m2_summer_nh+1))
    winter_sh_ind = (months >= m1_summer_nh) & (months <= m2_summer_nh)
    edd_winter_nh = edd_monthly[winter_nh_ind,:lat_cutoff,:].resample(time="YS").sum(dim="time")
    edd_winter_sh = edd_monthly[winter_sh_ind,lat_cutoff:,:].resample(time="YS").sum(dim="time")
    edd_winter = (xr.concat([edd_winter_nh,edd_winter_sh],dim="lat"))
    edd_winter_flip = flip_lon_tll(edd_winter[:,::-1,:])
    edd_winter_region = xr_region_average(edd_winter_flip,y1_edd,y2_edd,"YS",
                                      shapes,id_shp,idnums,True,pop_flip_halfdegree)

    # add to dataset and output

    ds = xr.Dataset({"edd_consecutive":(["region","time"],edd_annual_region),
                    "edd_non_consecutive":(["region","time"],edd_annual_notcons_region),
                    "edd_summer":(["region","time"],edd_summer_region),
                    "edd_winter":(["region","time"],edd_winter_region)},
                    coords={"region":(["region"],edd_annual_region.coords["region"]),
                            "time":(["time"],edd_annual_region.coords["time"])})
                    #"edd_seasonal":(["month_window","region","time"],edd_seas_region),
                    #"window_codes":(["month_window"],month_codes)
                    #"month_window":(["month_window"],edd_seas_region.coords["month_window"])
    ds.attrs["creation_date"] = str(datetime.datetime.now())
    ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
    ds.attrs["variable_description"] = "Extreme degree days at "+str(t)+"th percentile threshold, avg across 1st admin regions"
    ds.attrs["created_from"] = os.getcwd()+"/Calculate_Regional_T_EDD.py"

    fname_out = loc_edd_out+"ERA5_extreme_degree_days_"+threshold_type.lower()+"_percentile"+str(t)+"_adm1_region_"+str(y1_edd)+"-"+str(y2_edd)+".nc"
    ds.to_netcdf(fname_out,mode="w")
    print(fname_out,flush=True)
