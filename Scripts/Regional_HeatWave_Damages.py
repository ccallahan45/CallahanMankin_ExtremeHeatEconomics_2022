# Historical economic damages due to changes in heat waves in subnational regions
# also including changes in mean temperature and variability
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

## this takes a few hours
## on an HPC system
## would not recommend running on a local machine

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
from scipy import signal, stats
from functools import reduce

# Data locations
loc_panel = "../Data/Panel/"
loc_regression = "../Data/RegressionResults/"
loc_shp = "../Data/Shapefile/"
loc_cmip6_tx = "../Data/CMIP6/RegionTx/"
loc_cmip6_t = "../Data/CMIP6/RegionT/"
loc_cmip6_tvar = "../Data/CMIP6/RegionTvar/"
loc_custom_gdp = "../Data/Predicted_Regional_GDP/"
loc_irf = "../Data/IRFs/"
loc_out = "../Data/Damages/"

# suppress annoying warning for mean of empty slice
import warnings
warnings.filterwarnings("ignore",message="Mean of empty",category=RuntimeWarning)

# Years
y1_cmip6 = 1950
y2_cmip6 = 2020
y1 = 1979
y2 = 2016
years_cmip6 = np.arange(y1_cmip6,y2_cmip6+1,1)
years = np.arange(y1,y2+1,1)
y1_income = 1993 # second year of nightlight data (we lose the first year calculating growth)
y2_income = 2013 # last year of nightlight data
years_income = np.arange(y1_income,y2_income+1,1)

# read panel data - need T/Tx
print("reading miscellaneous obs data",flush=True)
edd_type = "month"
extr = "tx5d"
panel = pd.read_csv(loc_panel+"extremes_growth_panel_"+edd_type+"edd_"+str(y1)+"-"+str(y2)+".csv",index_col=0)
panel["iso"] = [x[0:3] for x in panel.region.values]

## read predicted regional GDPpc
regional_gdppc_ds = xr.open_dataset(loc_custom_gdp+"adm1_subnational_constant2010_gdp_percapita_predicted_"+str(y1_income-1)+"-"+str(y2_income)+".nc")
obs_gpc = regional_gdppc_ds.data_vars["gdp_per_capita"] # region x time x uncertainty
gpc_uncert = obs_gpc.coords["uncertainty"].values


# calculate growth
obs_gr1 = obs_gpc.diff(dim="time",n=1)
obs_frac_growth = obs_gr1/(obs_gpc.loc[:,:(y2_income-1),:].values)
growth_nans = xr.DataArray(np.full((len(np.unique(obs_gpc.region.values)),1,len(gpc_uncert)),np.nan),
                           coords=[np.unique(obs_gpc.region.values),[y1_income-1],gpc_uncert],
                           dims=["region","time","uncertainty"])
obs_gr = xr.concat([growth_nans,obs_frac_growth],dim="time")

# convert other stuff to xarray data-arrays
# for tx t and population
obs_tx_pivot = panel.loc[:,["region","time",extr]].pivot(index="region",columns="time")
obs_tx = xr.DataArray(obs_tx_pivot.values,
                    coords=[np.unique(panel.region.values),
                            np.unique(panel.time.values)],
                    dims=["region","time"])

obs_t_pivot = panel.loc[:,["region","time","t"]].pivot(index="region",columns="time")
obs_t = xr.DataArray(obs_t_pivot.values,
                    coords=[np.unique(panel.region.values),
                            np.unique(panel.time.values)],
                    dims=["region","time"])

obs_tvar_pivot = panel.loc[:,["region","time","var"]].pivot(index="region",columns="time")
obs_tvar = xr.DataArray(obs_tvar_pivot.values,
                    coords=[np.unique(panel.region.values),
                            np.unique(panel.time.values)],
                    dims=["region","time"])

obs_ann_pivot = panel.loc[:,["region","time","seas"]].pivot(index="region",columns="time")
obs_ann = xr.DataArray(obs_ann_pivot.values,
                    coords=[np.unique(panel.region.values),
                            np.unique(panel.time.values)],
                    dims=["region","time"])

obs_pop_pivot = panel.loc[:,["region","time","population"]].pivot(index="region",columns="time")
obs_pop = xr.DataArray(obs_pop_pivot.values,
                    coords=[np.unique(panel.region.values),
                            np.unique(panel.time.values)],
                    dims=["region","time"])


## read impulse-response functions based on the distributed lag regression coefficients
print("loading impulse-response functions based on the regression results",flush=True)
nlag = 5
boot_type = "region"

# extreme heat
tx_irf_ds = xr.open_dataset(loc_irf+extr+"_impulse-response_coefficients_bootstrap_"+boot_type+"_lag"+str(nlag)+".nc")
tx_coef_irf_final = tx_irf_ds.coef_main
tx_int_irf_final = tx_irf_ds.coef_interact
reg_boot = tx_coef_irf_final.reg_boot
convergence_year_tx = np.amax(tx_coef_irf_final.lag.values)

# mean temperature
t_irf_ds = xr.open_dataset(loc_irf+"temperature_impulse-response_coefficients_"+extr+"_model_bootstrap_"+boot_type+"_lag"+str(nlag)+".nc")
t_coef_irf_final = t_irf_ds.coef_main
t_int_irf_final = t_irf_ds.coef_interact
convergence_year_t = np.nan # persistent effects


# temperature variability
tvar_irf_ds = xr.open_dataset(loc_irf+"variability_impulse-response_coefficients_"+extr+"_model_bootstrap_"+boot_type+"_lag"+str(nlag)+".nc")
tvar_coef_irf_final = tvar_irf_ds.coef_main
tvar_int_irf_final = tvar_irf_ds.coef_interact
convergence_year_tvar = np.amax(tvar_coef_irf_final.lag.values)


# read CMIP6 data
print("reading CMIP6 data",flush=True)

vrs = [extr,"t","tvar"]
ssp = "ssp245"

for v in vrs:
    print(v,flush=True)
    if v==extr:
        loc_in = loc_cmip6_tx
    elif v=="t":
        loc_in = loc_cmip6_t
    elif v=="tvar":
        loc_in = loc_cmip6_tvar

    # and leave out FGOALS because we want to go up to 2020
    files_hist = np.array([x for x in sorted(os.listdir(loc_in)) if ("historical-"+ssp in x)&(str(y1_cmip6)+"-"+str(y2_cmip6) in x)&("FGOALS" not in x)])
    filenames_hist = np.array([loc_in+x for x in files_hist])
    if v==vrs[0]:
        models = np.array([x.split("_")[0]+"_"+x.split("_")[1] for x in files_hist if 'FGOALS' not in x]) # avoid FGOALS
        cmip6_vars_hist = xr.DataArray(np.full((len(vrs),len(models),len(obs_t.region.values),len(years_cmip6)),np.nan),
                                        coords=[vrs,models,obs_t.region.values,years_cmip6],
                                        dims=["variable","model","region","time"])
        cmip6_vars_histnat = xr.DataArray(np.full((len(vrs),len(models),len(obs_t.region.values),len(years_cmip6)),np.nan),
                                        coords=[vrs,models,obs_t.region.values,years_cmip6],
                                        dims=["variable","model","region","time"])
        print(models,flush=True)
        print(len(models),flush=True)

    files_nat = np.array([x for x in sorted(os.listdir(loc_in)) if ("historical-nat" in x)&(str(y1_cmip6)+"-"+str(y2_cmip6) in x)&("FGOALS" not in x)])
    filenames_nat = np.array([loc_in+x for x in files_nat])
    cmip6_hist_data = xr.open_mfdataset(filenames_hist,combine="nested",concat_dim="model").data_vars[v].load()
    cmip6_histnat_data = xr.open_mfdataset(filenames_nat,combine="nested",concat_dim="model").data_vars[v].load()
    cmip6_hist_data.coords["model"] = models
    cmip6_histnat_data.coords["model"] = models
    cmip6_vars_hist.loc[v,:,:,:] = cmip6_hist_data.loc[:,obs_t.region,:].values
    cmip6_vars_histnat.loc[v,:,:,:] = cmip6_histnat_data.loc[:,obs_t.region,:].values
    del([cmip6_hist_data,cmip6_histnat_data])

modelnames = xr.DataArray(np.array([x.split("_")[0] for x in models]),
                            coords=[models],dims=["model"])
mdlwgts = xr.DataArray(np.array([1.0/sum(modelnames==x) for x in modelnames]),
                    coords=[models],dims=["model"])
mdl_p = mdlwgts/np.sum(mdlwgts)

# smooth to eliminate random variation
smth = 15 # 0
if smth != 0:
    if smth<=(((y2_cmip6-y2_income)*2)+1):
        ctr = True
    else:
        ctr = False
else:
    ctr = False

if smth != 0:
    cmip6_hist_smooth = cmip6_vars_hist.rolling(time=smth,min_periods=smth,center=ctr).mean()
    cmip6_histnat_smooth = cmip6_vars_histnat.rolling(time=smth,min_periods=smth,center=ctr).mean()
else:
    cmip6_hist_smooth = cmip6_vars_hist*1.0
    cmip6_histnat_smooth = cmip6_vars_histnat*1.0
del(cmip6_vars_hist)
del(cmip6_vars_histnat)

# calculate counterfactuals
cmip6_diff = cmip6_hist_smooth - cmip6_histnat_smooth
del([cmip6_hist_smooth,cmip6_histnat_smooth])
cf_tx = obs_tx - cmip6_diff.sel(variable=extr)
cf_t = obs_t - cmip6_diff.sel(variable="t")
cf_tvar = obs_tvar - cmip6_diff.sel(variable="tvar")
del(cmip6_diff)


## now establish monte carlo/uncertainty distributions
## uncertainty from climate models and regression bootstraps
## and predicted GDP realizations
n_mc = 10000
mc_ind_mdl = np.zeros(n_mc) # climate model
mc_ind_reg = np.zeros(n_mc) # regression coefficient
mc_ind_gpc = np.zeros(n_mc) # predicted GDP realization
uncertainty = np.arange(1,n_mc+1,1)

print("constructing monte carlo parameter samples",flush=True)
np.random.seed(108)
for n in uncertainty:
    if (n==1)|(np.mod(n,n_mc/10.)==0):
        print(n)
    # climate models -- inversely weight models by number of realizations per model
    mc_ind_mdl[n-1] = int(np.random.choice(np.arange(0,len(models),1),size=1,p=mdl_p.values))
    # regression bootstrap ind
    mc_ind_reg[n-1] = int(np.random.choice(np.arange(0,len(reg_boot),1),size=1))
    # predicted GDP realization
    mc_ind_gpc[n-1] = int(np.random.choice(np.arange(0,len(gpc_uncert),1),size=1))

## calculate damages!

print("calculating damages",flush=True)

## create useful functions for growth calculations
def create_growth_arrays(dg,region,time):
    cds = [time,region]
    dms = ["time","region"]
    cf_gdp = xr.DataArray(np.full(dg.values.shape,np.nan),
                         coords=cds,dims=dms)
    cf_growth = xr.DataArray(np.full(dg.values.shape,np.nan),
                         coords=cds,dims=dms)
    return([cf_gdp,cf_growth])

def calc_counterfactual_gdp(orig_growth,orig_gpc,delta_growth,region,time):
    yrs = time
    cf_gpc, cf_growth = create_growth_arrays(delta_growth,region,time)
    cf_gr = orig_growth.loc[:,yrs].transpose("time","region") + delta_growth
    cf_growth[:,:] = cf_gr.transpose("time","region").values
    for yy in np.arange(1,len(yrs),1):
        if yy == 1:
            cf_gdp_year = orig_gpc.loc[:,yrs[yy-1]]
        else:
            cf_gdp_year = cf_gpc.loc[yrs[yy-1],:]
        cf_gpc.loc[yrs[yy],:] = cf_gdp_year+(cf_growth.loc[yrs[yy],:]*cf_gdp_year)
    #return([cf_gpc, cf_growth])
    cf_gpc = cf_gpc.transpose("region","time")
    return(cf_gpc)


# get list of regions with continuous data
len_yr_income = len(years_income)
obs_gr_notnan = ~np.isnan(obs_gr.loc[:,y1_income:y2_income,1].values)
obs_gpc_notnan = ~np.isnan(obs_gpc.loc[:,y1_income:y2_income,1].values)
len_growth_notnan = np.sum(obs_gr_notnan,axis=1)
len_gpc_notnan = np.sum(obs_gpc_notnan,axis=1)
regions = obs_gpc.region.values
region_continuous = regions[(len_growth_notnan==len_yr_income)&(len_gpc_notnan==len_yr_income)]
obs_gr_continuous = obs_gr.loc[region_continuous,y1_income:y2_income,:]
obs_gpc_continuous = obs_gpc.loc[region_continuous,y1_income:y2_income,:]
obs_pop_continuous = obs_pop.loc[region_continuous,y1_income:y2_income]
obs_gdp_continuous = obs_gpc.loc[region_continuous,y1_income:y2_income,:]*obs_pop.loc[region_continuous,y1_income:y2_income]

#print(obs_gdp_continuous[["BRA" in x for x in obs_gdp_continuous.region.values],:,:].sel(time=2010).mean(dim=["region","uncertainty"]))
#print(obs_gdp_continuous[["IND" in x for x in obs_gdp_continuous.region.values],:,:].sel(time=2010).mean(dim=["region","uncertainty"]))
#sys.exit()

## set up income change arrays and functions
dgdp_pct_tx = xr.DataArray(np.full((len(region_continuous),len(years_income),len(uncertainty)),np.nan),
                            coords=[region_continuous,years_income,uncertainty],
                            dims=["region","time","uncertainty"])
dgdp_absolute_tx = xr.DataArray(np.full((len(region_continuous),len(years_income),len(uncertainty)),np.nan),
                                coords=[region_continuous,years_income,uncertainty],
                                dims=["region","time","uncertainty"])
dgdp_pct_tmean = xr.DataArray(np.full((len(region_continuous),len(years_income),len(uncertainty)),np.nan),
                            coords=[region_continuous,years_income,uncertainty],
                            dims=["region","time","uncertainty"])
dgdp_absolute_tmean = xr.DataArray(np.full((len(region_continuous),len(years_income),len(uncertainty)),np.nan),
                                coords=[region_continuous,years_income,uncertainty],
                                dims=["region","time","uncertainty"])
dgdp_pct_tvar = xr.DataArray(np.full((len(region_continuous),len(years_income),len(uncertainty)),np.nan),
                            coords=[region_continuous,years_income,uncertainty],
                            dims=["region","time","uncertainty"])
dgdp_absolute_tvar = xr.DataArray(np.full((len(region_continuous),len(years_income),len(uncertainty)),np.nan),
                                coords=[region_continuous,years_income,uncertainty],
                                dims=["region","time","uncertainty"])
dgdp_pct_all = xr.DataArray(np.full((len(region_continuous),len(years_income),len(uncertainty)),np.nan),
                            coords=[region_continuous,years_income,uncertainty],
                            dims=["region","time","uncertainty"])
dgdp_absolute_all = xr.DataArray(np.full((len(region_continuous),len(years_income),len(uncertainty)),np.nan),
                                coords=[region_continuous,years_income,uncertainty],
                                dims=["region","time","uncertainty"])

# check time elapsed
import time
start = time.time()

## loop through MC realizations

for n in uncertainty:
    #if (n==1)|(np.mod(n,n_mc/10.)==0):
    print(n,flush=True)
    mdl_ind = int(mc_ind_mdl[n-1])
    reg_boot_ind = int(mc_ind_reg[n-1])
    gpc_uncert_ind = int(mc_ind_gpc[n-1])

    # obs and counterfactual Tx and income
    obs_tx_n = obs_tx.loc[region_continuous,y1_income:y2_income]
    obs_t_n = obs_t.loc[region_continuous,y1_income:y2_income]
    obs_tvar_n = obs_tvar.loc[region_continuous,y1_income:y2_income]
    cf_tx_n = cf_tx[:,:,mdl_ind].loc[region_continuous,y1_income:y2_income]
    cf_t_n = cf_t[:,:,mdl_ind].loc[region_continuous,y1_income:y2_income]
    cf_tvar_n = cf_tvar[:,:,mdl_ind].loc[region_continuous,y1_income:y2_income]
    obs_ann_n = obs_ann.loc[region_continuous,y1_income:y2_income]
    obs_gpc_n = obs_gpc_continuous[:,:,gpc_uncert_ind]
    obs_gr_n = obs_gr_continuous[:,:,gpc_uncert_ind]

    # expand coefficients -- region x lag
    coefs_tx_main_xr2 = tx_coef_irf_final[reg_boot_ind,:].expand_dims(region=region_continuous)
    coefs_tx_int_xr2 = tx_int_irf_final[reg_boot_ind,:].expand_dims(region=region_continuous)
    coefs_t_main_xr2 = t_coef_irf_final[reg_boot_ind,:].expand_dims(region=region_continuous)
    coefs_t_int_xr2 = t_int_irf_final[reg_boot_ind,:].expand_dims(region=region_continuous)
    coefs_tvar_main_xr2 = tvar_coef_irf_final[reg_boot_ind,:].expand_dims(region=region_continuous)
    coefs_tvar_int_xr2 = tvar_int_irf_final[reg_boot_ind,:].expand_dims(region=region_continuous)

    # calculate marginal effects
    me_tx_1 = coefs_tx_main_xr2 + coefs_tx_int_xr2*cf_t_n # counterfactual mean temp
    me_tx_2 = coefs_tx_main_xr2 + coefs_tx_int_xr2*obs_t_n # obs t

    me_t_1 = coefs_t_main_xr2 + coefs_t_int_xr2*cf_tx_n # counterfactual tx
    me_t_2 = coefs_t_main_xr2 + coefs_t_int_xr2*obs_tx_n # obs tx

    x1 = me_t_1[["BRA" in x for x in me_t_1.region.values],:,:].sum(dim="lag")
    x2 = me_t_2[["BRA" in x for x in me_t_2.region.values],:,:].sum(dim="lag")
    x3 = cf_tx_n[["BRA" in x for x in cf_tx_n.region.values],:] - obs_tx_n[["BRA" in x for x in obs_tx_n.region.values],:]
    #print((x1 - x2).mean(dim="time")*100)
    #print(x3.mean(dim="time"))
    #sys.exit()

    me_tvar_1 = coefs_tvar_main_xr2 + coefs_tvar_int_xr2*obs_ann_n # obs ann cycle
    me_tvar_2 = coefs_tvar_main_xr2 + coefs_tvar_int_xr2*obs_ann_n # obs ann cycle

    # lag 0 and then add up over lags
    # make sure that things end up summing to zero for tx and tvar

    # first, for tx
    delta_growth_tx = (me_tx_1.loc[:,0,:]*cf_tx_n) - (me_tx_2.loc[:,0,:]*obs_tx_n)
    for l in np.arange(1,convergence_year_tx+1,1):
        delta_tx = (me_tx_1.loc[:,l,:].shift(time=l)*cf_tx_n.shift(time=l)) - (me_tx_2.loc[:,l,:].shift(time=l)*obs_tx_n.shift(time=l))
        delta_growth_tx = delta_growth_tx + delta_tx.where(~np.isnan(delta_tx),0.0)

    # next, for tmean
    delta_growth_t = (me_t_1.loc[:,0,:]*cf_t_n) - (me_t_2.loc[:,0,:]*obs_t_n)
    for l in np.arange(1,np.amax(me_t_1.lag.values)+1,1):
        delta_t = (me_t_1.loc[:,l,:].shift(time=l)*cf_t_n.shift(time=l)) - (me_t_2.loc[:,l,:].shift(time=l)*obs_t_n.shift(time=l))
        delta_growth_t = delta_growth_t + delta_t.where(~np.isnan(delta_t),0.0)

    # finally, tvar
    delta_growth_tvar = (me_tvar_1.loc[:,0,:]*cf_tvar_n) - (me_tvar_2.loc[:,0,:]*obs_tvar_n)
    for l in np.arange(1,convergence_year_tvar+1,1):
        delta_tvar = (me_tvar_1.loc[:,l,:].shift(time=l)*cf_tvar_n.shift(time=l)) - (me_tvar_2.loc[:,l,:].shift(time=l)*obs_tvar_n.shift(time=l))
        delta_growth_tvar = delta_growth_tvar + delta_tvar.where(~np.isnan(delta_tvar),0.0)

    # calculate change in income from change in growth
    cf_gpc_tx_n = calc_counterfactual_gdp(obs_gr_n,obs_gpc_n,delta_growth_tx.transpose("time","region"),
                                            region_continuous,years_income)
    cf_gpc_t_n = calc_counterfactual_gdp(obs_gr_n,obs_gpc_n,delta_growth_t.transpose("time","region"),
                                            region_continuous,years_income)
    cf_gpc_tvar_n = calc_counterfactual_gdp(obs_gr_n,obs_gpc_n,delta_growth_tvar.transpose("time","region"),
                                            region_continuous,years_income)

    # all three factors together
    delta_growth_all = delta_growth_tx.transpose("time","region")+delta_growth_t.transpose("time","region")+delta_growth_tvar.transpose("time","region")
    cf_gpc_all_n = calc_counterfactual_gdp(obs_gr_n,obs_gpc_n,delta_growth_all,region_continuous,years_income)

    # add to overall array
    dgdp_pct_tx.loc[:,:,n] = 100*(obs_gpc_n - cf_gpc_tx_n)/cf_gpc_tx_n
    dgdp_absolute_tx.loc[:,:,n] = (obs_gpc_n - cf_gpc_tx_n)*obs_pop_continuous

    dgdp_pct_tmean.loc[:,:,n] = 100*(obs_gpc_n - cf_gpc_t_n)/cf_gpc_t_n
    dgdp_absolute_tmean.loc[:,:,n] = (obs_gpc_n - cf_gpc_t_n)*obs_pop_continuous

    dgdp_pct_tvar.loc[:,:,n] = 100*(obs_gpc_n - cf_gpc_tvar_n)/cf_gpc_tvar_n
    dgdp_absolute_tvar.loc[:,:,n] = (obs_gpc_n - cf_gpc_tvar_n)*obs_pop_continuous

    dgdp_pct_all.loc[:,:,n] = 100*(obs_gpc_n - cf_gpc_all_n)/cf_gpc_all_n
    dgdp_absolute_all.loc[:,:,n] = (obs_gpc_n - cf_gpc_all_n)*obs_pop_continuous

    if np.mod(n,100)==0:
        end = time.time()
        print((end-start)/60.,flush=True)

dgdp_pct_tx_mean = dgdp_pct_tx.mean(dim="time")
dgdp_pct_tx_bool = (dgdp_pct_tx_mean < 0).mean(dim="uncertainty")
dgdp_bool_damages = dgdp_pct_tx_bool[dgdp_pct_tx_mean.mean(dim="uncertainty") < 0]
ndamage = len(dgdp_bool_damages)

print(len(dgdp_bool_damages[dgdp_bool_damages>0.6])/ndamage)
print(len(dgdp_bool_damages[dgdp_bool_damages>0.7])/ndamage)
print(len(dgdp_bool_damages[dgdp_bool_damages>0.8])/ndamage)
print(len(dgdp_bool_damages[dgdp_bool_damages>0.9])/ndamage)

end = time.time()
print((end-start)/60.,flush=True)

vrs_out = [extr,"temperature","variability","allvariables"]
print("writing out",flush=True)

for v in vrs_out:
    print(v,flush=True)

    if v==extr:
        gpc_change_pct = dgdp_pct_tx
        gdp_change = dgdp_absolute_tx
    elif v=="temperature":
        gpc_change_pct = dgdp_pct_tmean
        gdp_change = dgdp_absolute_tmean
    elif v=="variability":
        gpc_change_pct = dgdp_pct_tvar
        gdp_change = dgdp_absolute_tvar
    elif v=="allvariables":
        gpc_change_pct = dgdp_pct_all
        gdp_change = dgdp_absolute_all

    qs = np.array([0.005,0.025,0.05,0.165,0.17,0.5,0.83,0.835,0.95,0.975,0.995])
    gpc_change_pct_mean = gpc_change_pct.mean(dim="uncertainty")
    gpc_change_pct_std = gpc_change_pct.std(dim="uncertainty")

    # calc SNR and limit the global summing to regions with SNR > 1
    snr = np.abs(gpc_change_pct_mean.mean(dim="time"))/gpc_change_pct_std.mean(dim="time")
    gdp_change_forsum = gdp_change.where(snr>=1,0.0)
    gdp_change_mean = gdp_change.mean(dim="uncertainty")
    gdp_change_std = gdp_change.std(dim="uncertainty")
    gdp_change_cumulative = np.cumsum(gdp_change_forsum.sum(dim="region"),axis=0)
    gdp_change_cumulative_mean = gdp_change_cumulative.mean(dim="uncertainty")
    gdp_change_cumulative_std = gdp_change_cumulative.std(dim="uncertainty")
    gdp_change_cumulative_quantiles = gdp_change_cumulative.quantile(qs,dim="uncertainty")

    # now also include a version where all regions are included
    gdp_change_cumulative_all = np.cumsum(gdp_change.sum(dim="region"),axis=0)
    gdp_change_cumulative_all_mean = gdp_change_cumulative_all.mean(dim="uncertainty")
    gdp_change_cumulative_all_quantiles = gdp_change_cumulative_all.quantile(qs,dim="uncertainty")

    # pct of monte carlo runs
    gpc_change_pct_timemean = gpc_change_pct.mean(dim="time")
    gpc_change_positive_mcpct = (gpc_change_pct_timemean > 0).mean(dim="uncertainty")
    gpc_change_negative_mcpct = (gpc_change_pct_timemean < 0).mean(dim="uncertainty")

    # quantiles by 2010 income decile
    obs_gpc_2010_mean = obs_gpc_continuous.sel(time=2010).mean(dim="uncertainty")
    gpc_deciles = pd.qcut(obs_gpc_2010_mean.values,10,labels=False)
    gpc_change_pct.coords["decile"] = xr.DataArray(gpc_deciles,coords=[obs_gpc_2010_mean.region],dims=["region"])
    gpc_change_pct_mean_decile = gpc_change_pct.groupby("decile").mean(dim=["region","time"])

    # add to dataset and write out
    region_damage_ds = xr.Dataset({"gpc_change_pct_mean":(["region","time"],gpc_change_pct_mean),
                                    "gpc_change_pct_std":(["region","time"],gpc_change_pct_std),
                                    "gdp_change_mean":(["region","time"],gdp_change_mean),
                                    "gdp_change_std":(["region","time"],gdp_change_std),
                                    "gdp_change_cumulative_mean":(["time"],gdp_change_cumulative_mean),
                                    "gdp_change_cumulative_std":(["time"],gdp_change_cumulative_std),
                                    "gdp_change_cumulative_quantiles":(["quantile","time"],gdp_change_cumulative_quantiles),
                                    "gdp_change_cumulative_all_mean":(["time"],gdp_change_cumulative_all_mean),
                                    "gdp_change_cumulative_all_quantiles":(["quantile","time"],gdp_change_cumulative_all_quantiles),
                                    "gpc_change_pct_mean_decile":(["decile","uncertainty"],gpc_change_pct_mean_decile),
                                    "gpc_change_mc_pct_negative":(["region"],gpc_change_negative_mcpct),
                                    "gpc_change_mc_pct_positive":(["region"],gpc_change_positive_mcpct),
                                    "obs_gpc_mean":(["region","time"],obs_gpc_continuous.mean(dim="uncertainty")),
                                    "obs_gdp_mean":(["region","time"],obs_gdp_continuous.mean(dim="uncertainty"))},
                                coords={"region":(["region"],region_continuous),
                                        "time":(["time"],years_income),
                                        "quantile":(["quantile"],qs),
                                        "decile":(["decile"],gpc_change_pct_mean_decile.decile),
                                        "uncertainty":(["uncertainty"],gpc_change_pct_mean_decile.uncertainty)})

    region_damage_ds.attrs["creation_date"] = str(datetime.datetime.now())
    region_damage_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
    region_damage_ds.attrs["variable_description"] = "Subnational region income effects of changing extreme heat, mean temps, or temp variability"
    region_damage_ds.attrs["created_from"] = os.getcwd()+"/Regional_HeatWave_Damages.py"
    region_damage_ds.attrs["dims"] = "region, time, quantile"
    region_damage_ds.attrs["uncertainty"] = str(n_mc)+" monte carlo samples"
    region_damage_ds.attrs["inflation_adjustment"] = "constant 2010 dollars"

    if v == extr:
        fname_out = loc_out+"CMIP6_historical_region_"+v+"_income_damages_smooth"+str(smth)+"_"+str(y1_income)+"-"+str(y2_income)+".nc"
    else:
        fname_out = loc_out+"CMIP6_historical_region_"+v+"_income_damages_"+extr+"_model_smooth"+str(smth)+"_"+str(y1_income)+"-"+str(y2_income)+".nc"
    region_damage_ds.to_netcdf(fname_out,mode="w")
    print(fname_out,flush=True)


del([dgdp_pct_tx,dgdp_absolute_tx,dgdp_pct_tmean,dgdp_absolute_tmean])
del([dgdp_pct_tvar,dgdp_absolute_tvar,dgdp_pct_all,dgdp_absolute_all])



# now do the same thing just for tx without t change
print("running tx damages without temperature change",flush=True)

dgdp_pct_tx = xr.DataArray(np.full((len(region_continuous),len(years_income),len(uncertainty)),np.nan),
                            coords=[region_continuous,years_income,uncertainty],
                            dims=["region","time","uncertainty"])
dgdp_absolute_tx = xr.DataArray(np.full((len(region_continuous),len(years_income),len(uncertainty)),np.nan),
                                coords=[region_continuous,years_income,uncertainty],
                                dims=["region","time","uncertainty"])

# check time elapsed
import time
start = time.time()

## loop through MC realizations

for n in uncertainty:
    #if (n==1)|(np.mod(n,n_mc/10.)==0):
    print(n,flush=True)
    mdl_ind = int(mc_ind_mdl[n-1])
    reg_boot_ind = int(mc_ind_reg[n-1])
    gpc_uncert_ind = int(mc_ind_gpc[n-1])

    # obs and counterfactual Tx and income
    obs_tx_n = obs_tx.loc[region_continuous,y1_income:y2_income]
    obs_t_n = obs_t.loc[region_continuous,y1_income:y2_income]
    cf_tx_n = cf_tx[:,:,mdl_ind].loc[region_continuous,y1_income:y2_income]
    obs_gpc_n = obs_gpc_continuous[:,:,gpc_uncert_ind]
    obs_gr_n = obs_gr_continuous[:,:,gpc_uncert_ind]

    # expand coefficients -- region x lag
    coefs_tx_main_xr2 = tx_coef_irf_final[reg_boot_ind,:].expand_dims(region=region_continuous)
    coefs_tx_int_xr2 = tx_int_irf_final[reg_boot_ind,:].expand_dims(region=region_continuous)

    # calculate marginal effects
    me_tx_1 = coefs_tx_main_xr2 + coefs_tx_int_xr2*obs_t_n # obs_t
    me_tx_2 = coefs_tx_main_xr2 + coefs_tx_int_xr2*obs_t_n # obs t

    # lag 0 and then add up over lags
    # make sure that things end up summing to zero for tx
    delta_growth_tx = (me_tx_1.loc[:,0,:]*cf_tx_n) - (me_tx_2.loc[:,0,:]*obs_tx_n)
    for l in np.arange(1,convergence_year_tx+1,1):
        delta_tx = (me_tx_1.loc[:,l,:]*cf_tx_n.shift(time=l)) - (me_tx_2.loc[:,l,:]*obs_tx_n.shift(time=l))
        delta_growth_tx = delta_growth_tx + delta_tx.where(~np.isnan(delta_tx),0.0)


    # calculate change in income from change in growth
    cf_gpc_tx_n = calc_counterfactual_gdp(obs_gr_n,obs_gpc_n,delta_growth_tx.transpose("time","region"),
                                            region_continuous,years_income)

    # add to overall array
    dgdp_pct_tx.loc[:,:,n] = 100*(obs_gpc_n - cf_gpc_tx_n)/cf_gpc_tx_n
    dgdp_absolute_tx.loc[:,:,n] = (obs_gpc_n - cf_gpc_tx_n)*obs_pop_continuous

    if np.mod(n,100)==0:
        end = time.time()
        print((end-start)/60.,flush=True)

end = time.time()
print((end-start)/60.,flush=True)

gpc_change_pct = dgdp_pct_tx*1.0
gdp_change = dgdp_absolute_tx*1.0
del([dgdp_pct_tx,dgdp_absolute_tx])

qs = np.array([0.005,0.025,0.05,0.165,0.835,0.95,0.975,0.995])
gpc_change_pct_mean = gpc_change_pct.mean(dim="uncertainty")
gpc_change_pct_std = gpc_change_pct.std(dim="uncertainty")
gdp_change_mean = gdp_change.mean(dim="uncertainty")
gdp_change_std = gdp_change.std(dim="uncertainty")

# now also include a version where all regions are included
gdp_change_cumulative_all = np.cumsum(gdp_change.sum(dim="region"),axis=0)
gdp_change_cumulative_all_mean = gdp_change_cumulative_all.mean(dim="uncertainty")
gdp_change_cumulative_all_quantiles = gdp_change_cumulative_all.quantile(qs,dim="uncertainty")

# add to dataset and write out
region_damage_ds = xr.Dataset({"gpc_change_pct_mean":(["region","time"],gpc_change_pct_mean),
                                "gpc_change_pct_std":(["region","time"],gpc_change_pct_std),
                                "gdp_change_mean":(["region","time"],gdp_change_mean),
                                "gdp_change_cumulative_all_mean":(["time"],gdp_change_cumulative_all_mean),
                                "gdp_change_cumulative_all_quantiles":(["quantile","time"],gdp_change_cumulative_all_quantiles),
                                "obs_gpc_mean":(["region","time"],obs_gpc_continuous.mean(dim="uncertainty")),
                                "obs_gdp_mean":(["region","time"],obs_gdp_continuous.mean(dim="uncertainty"))},
                            coords={"region":(["region"],region_continuous),
                                    "time":(["time"],years_income),
                                    "quantile":(["quantile"],qs)})

region_damage_ds.attrs["creation_date"] = str(datetime.datetime.now())
region_damage_ds.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
region_damage_ds.attrs["variable_description"] = "Subnational region income effects of changing extreme heat without changing temperatures"
region_damage_ds.attrs["created_from"] = os.getcwd()+"/Regional_HeatWave_Damages.py"
region_damage_ds.attrs["dims"] = "region, time, quantile"
region_damage_ds.attrs["uncertainty"] = str(n_mc)+" monte carlo samples"
region_damage_ds.attrs["inflation_adjustment"] = "constant 2010 dollars"

fname_out = loc_out+"CMIP6_historical_region_"+extr+"_income_damages_constant_temp_smooth"+str(smth)+"_"+str(y1_income)+"-"+str(y2_income)+".nc"
region_damage_ds.to_netcdf(fname_out,mode="w")
print(fname_out,flush=True)
