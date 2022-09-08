# Relationship between extreme heat and economic growth
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

rm(list=ls())
# Libraries
library(ggplot2)
library(tidyr)
library(lfe)
library(dplyr)
library(lemon)
library(texreg)
library(cowplot)
library(gridExtra)
library(caret)
library(lmtest)

# locations

loc_data <- "../Data/Panel/"
loc_save_reg <- "..//Data/RegressionResults/"

# read in data
y1 <- 1979
y2 <- 2016
threshold_type <- "month"
panel_in <- read.csv(paste0(loc_data,"extremes_growth_panel_",threshold_type,"edd_",y1,"-",y2,".csv"))
panel_in %>% filter(time>=y1,time<=y2) %>% drop_na(growth) -> panel

# create some variables
panel$t2 <- (panel$t)**2
panel$p2 <- (panel$p)**2
panel$tmean2 <- (panel$tmean)**2
panel$edd <- panel$edd98
panel$seas2 <- (panel$seas)**2
panel$seas2_ann <- (panel$seas_ann)**2
panel$t2_summer <- (panel$t_summer)**2
panel$t2_winter <- (panel$t_winter)**2

# five year blocks by country
panel %>% rowwise() %>% 
  mutate(block=round((time+2)/5)*5,
         yr_iso=paste0(iso,"_",time)) %>%
  mutate(year_block=paste0(iso,"_",block)) -> panel

# lags
vars_to_lag <- c("t","t2","edd","var","p","p2","growth",
                 "tx7d","t_summer","seas_ann","txx","tx5d",
                 "luminosity")
for (v in c(1:length(vars_to_lag))){
  var <- vars_to_lag[v]
  print(var)
  for (l in c(1:10)){
    panel %>% group_by(region) %>% 
      mutate(!!paste(var,"_lag",l,sep="") := lag((!!as.name(var)),l)) -> panel
  }
}


################ 
#### How representative is the sample?
################

max(panel_in$t,na.rm=T)
max(panel_in[!is.na(panel_in$growth),]$t,na.rm=T)

yrs_pop_frac <- numeric(length(y1:y2))
yrs <- c(y1:y2)
for (y in c(1:length(yrs))){
  yy <- yrs[y]
  panel_y <- panel_in[panel_in$time==yy,]
  panel_y_all_pop <- sum(panel_y$population,na.rm=T)
  panel_y_sample_pop <- sum(panel_y[!is.na(panel_y$growth),]$population,na.rm=T)
  yrs_pop_frac[y] <- panel_y_sample_pop/panel_y_all_pop
}
mean(yrs_pop_frac,na.rm=T)


################ 
#### Table for clustering/trends
################
  

## model 1: original
panel %>% filter(t!=0) -> dat # when t is exactly 0 it's an error
dat$growth <- dat$growth*100
mdl1 <- felm(as.formula("growth ~ t + t2 + tx5d + tx5d:t + var + var:seas + p | region + time | 0 | region"),
             data=dat)

## model 2: precip squared
panel %>% filter(t!=0) -> dat # when t is exactly 0 it's an error
dat$growth <- dat$growth*100
mdl2 <- felm(as.formula("growth ~ t + t2 + tx5d + tx5d:t + var + var:seas + p + p2 | region + time | 0 | region"),
             data=dat)

## model 3: clustering
#dat$tsrev <- 30 - dat$t_summer
mdl3 <- felm(as.formula("growth ~ t + t2 + tx5d + tx5d:t + var + var:seas + p | region + time | 0 | iso"),
             data=dat)


## model 4: linear time trends
dat$time1 <- dat$time - 1979
# code here stolen from burke davis diffenbaugh, nature, 2018
ids <- as.vector(unique(dat$region))
for (y in 1:length(ids)){				
  dat[,paste('timelin',ids[y],sep="")] <- as.numeric(dat$region==ids[y])*dat$time1
}
trend.lin <- names(dat)[substr(names(dat), 1, 7)=="timelin"] %>%
  paste0(" + ") %>% as.list %>% do.call(paste0, .)

mdl4 <- felm(as.formula(paste0("growth ~ t + t2 + tx5d + tx5d:t + ",trend.lin,"var + var:seas + p | region + time | 0 | region")),
             data=dat)

## write out table
texreg(list(mdl1,mdl2,mdl3,mdl4),digits=4,
       omit.coef="timelin",stars=c(0.001,0.01,0.05))




################ 
#### Table for different interaction variables
################



## model 1: original
panel %>% filter(t!=0) -> dat # when t is exactly 0 it's an error
dat$growth <- dat$growth*100
mdl1 <- felm(as.formula("growth ~ t + t2 + tx5d + tx5d:t + var + var:seas + p | region + time | 0 | region"),
             data=dat)

## model 2: mean temperature
panel %>% filter(t!=0) -> dat # when t is exactly 0 it's an error
dat$growth <- dat$growth*100
mdl2 <- felm(as.formula("growth ~ t + t2 + tx5d + tx5d:tmean + var + var:seas + p | region + time | 0 | region"),
             data=dat)

## model 3: income
panel %>% filter(t!=0) -> dat # when t is exactly 0 it's an error
dat$growth <- dat$growth*100
dat %>% group_by(region) %>% mutate(mean_gpc=mean(lgdppc_2010,na.rm=T)) -> dat
mdl3 <- felm(as.formula("growth ~ t + t2 + tx5d + tx5d:mean_gpc + var + var:seas + p | region + time | 0 | region"),
             data=dat)

## model 4: income and temp
panel %>% filter(t!=0) -> dat # when t is exactly 0 it's an error
dat$growth <- dat$growth*100
dat %>% group_by(region) %>% mutate(mean_gpc=mean(lgdppc_2010,na.rm=T)) -> dat
mdl4 <- felm(as.formula("growth ~ t + t2 + tx5d + tx5d:mean_gpc + tx5d:t + var + var:seas + p | region + time | 0 | region"),
             data=dat)

## model 5: PDSI
panel %>% filter(t!=0) -> dat # when t is exactly 0 it's an error
dat$growth <- dat$growth*100
mdl5 <- felm(as.formula("growth ~ t + t2 + tx5d + tx5d:pdsi + var + var:seas + p | region + time | 0 | region"),
             data=dat)

## model 6: PDSI and temp
panel %>% filter(t!=0) -> dat # when t is exactly 0 it's an error
dat$growth <- dat$growth*100
mdl6 <- felm(as.formula("growth ~ t + t2 + tx5d + tx5d:pdsi + tx5d:t + var + var:seas + p | region + time | 0 | region"),
             data=dat)


## write out table
texreg(list(mdl1,mdl2,mdl3,mdl4,mdl5,mdl6),digits=4,
       stars=c(0.001,0.01,0.05))



################ 
#### Within vs. across variation in T
################

print(sd(panel$t,na.rm=T))
panel %>% 
  group_by(region) %>%
  summarize(tmean=sd(t)) %>%
  select(region,tmean) -> sd_within
print(mean(sd_within$tmean,na.rm=T))


################ 
#### Likelihood ratio test 
################


panel %>% filter(t!=0,!is.na(growth)) -> dat
mdl_t <- felm(as.formula("growth ~ t + t2 + var + var:seas | region + time | 0 | region"),data=dat)
mdl_tx <- felm(as.formula("growth ~ t + t2 + tx5d + tx5d:t + var + var:seas | region + time | 0 | region"),data=dat)
lrtest(mdl_t,mdl_tx)




################ 
#### Is variability intrinsically damaging?
################

panel %>% filter(t!=0,!is.na(growth)) -> dat

# scaling
dat %>% group_by(region) %>%
  summarize(tx_sd = sd(tx5d,na.rm=T),
            t_sd = sd(t,na.rm=T),
            var_sd = sd(var,na.rm=T)) -> sd_dat
tx_scaling <- mean(sd_dat$tx_sd,na.rm=T)
t_scaling <- mean(sd_dat$t_sd,na.rm=T)
var_scaling <- mean(sd_dat$var_sd,na.rm=T)


mdl_tx <- felm(as.formula("growth ~ t + t2 + tx5d + tx5d:t + var + var:seas | region + time | 0 | region"),data=dat)
print(summary(mdl_tx))
print(coef(summary(mdl_tx))["var","Estimate"]*var_scaling*100)
mdl_var <- felm(as.formula("growth ~ t + t2 + var + var:seas | region + time | 0 | region"),data=dat)
print(summary(mdl_var))
print(coef(summary(mdl_var))["var","Estimate"]*var_scaling*100)



################ 
#### Bootstrap coefficients across tx metrics
################

metrics <- c("tx5d","txx","tx3d","tx7d","tx15d","tmonx")
#metrics <- c("tx15d","tmonx")

nboot <- 1000
fe <- "region + time"
cl <- "0"
i <- "t" # interaction
panel %>% filter(t!=0,!is.na(growth)) -> dat

# bootstrap by region
for (mm in c(1:length(metrics))){
  extr <- metrics[mm]
  form <- as.formula(paste0("growth ~ t + t2 + ",extr," + ",extr,":",i," + var + var:seas + p | ",fe," | 0 | ",cl))
  print(extr)
  
  # set up data
  mdl_df <- data.frame("boot"=c(1:nboot),
                       "coef_main"=numeric(nboot),
                       "coef_interact"=numeric(nboot),
                       "aic"=numeric(nboot),
                       "adj_r2"=numeric(nboot))
  coefs_t <- data.frame("boot"=c(1:nboot),
                       "coef_t"=numeric(nboot),
                       "coef_t2"=numeric(nboot))
  coefs_var <- data.frame("boot"=c(1:nboot),
                        "coef_var"=numeric(nboot),
                        "coef_interact"=numeric(nboot))
  
  # loop through bootstrap iterations
  set.seed(120)
  for (n in c(1:nboot)){
    print(n)
    
    # bootstrap by region
    ids <- unique(dat$region)
    regions_boot <- sample(ids,size=length(ids),replace=T)
    df_boot <- sapply(regions_boot, function(x) which(dat[,'region']==x))
    data_boot <- dat[unlist(df_boot),]
    
    # run model
    mdl <- felm(form,data=data_boot)
    
    # save all the different coefficients
    mdl_df[n,"coef_main"] <- coef(summary(mdl))[extr,"Estimate"]
    if (i=="t"){
      mdl_df[n,"coef_interact"] <- coef(summary(mdl))[paste0(i,":",extr),"Estimate"]
    } else {
      mdl_df[n,"coef_interact"] <- coef(summary(mdl))[paste0(extr,":",i),"Estimate"]
    }
    mdl_df[n,"aic"] <- AIC(mdl)
    mdl_df[n,"adj_r2"] <- summary(mdl)$adj.r.squared
    
    # other coefficients
    coefs_t[n,"coef_t"] <- coef(summary(mdl))["t","Estimate"]
    coefs_t[n,"coef_t2"] <- coef(summary(mdl))["t2","Estimate"]
    coefs_var[n,"coef_var"] <- coef(summary(mdl))["var","Estimate"]
    coefs_var[n,"coef_interact"] <- coef(summary(mdl))["var:seas","Estimate"]
    
  }
  # write out when done with iterations
  write.csv(mdl_df,paste0(loc_save_reg,extr,"_coefs_bootstrap_contemporaneous.csv"))
  write.csv(coefs_t,paste0(loc_save_reg,"temperature_coefs_",extr,"_model_bootstrap_contemporaneous.csv"))
  write.csv(coefs_var,paste0(loc_save_reg,"variability_coefs_",extr,"_model_bootstrap_contemporaneous.csv"))
  print(paste0(loc_save_reg,extr,"_coefs_bootstrap_contemporaneous.csv"))
}




################ 
#### Distributed lag model
################

nboot <- 1000

fe <- "region + time"
cl <- "0"
#panel %>% filter(edd<50) -> dat
panel %>% filter(t!=0,!is.na(growth)) -> dat
dat$time1 <- dat$time - 1979
# code here stolen from burke davis diffenbaugh, nature, 2018
#ids <- as.vector(unique(dat$region))
#for (y in 1:length(ids)){				
#  dat[,paste('timelin',ids[y],sep="")] <- as.numeric(dat$region==ids[y])*dat$time1
#}
#trend.lin <- names(dat)[substr(names(dat), 1, 7)=="timelin"] %>%
#  paste0(" + ") %>% as.list %>% do.call(paste0, .)
trends <- "" #trend.lin # ""

extr <- "tx5d"
i <- "t"
bootstrap_type <- "country" #region

# loop through lags
bootstrap <- TRUE
lags_array <- c(5) #c(1:5)
## also estimate ARDL model alongside the other models

for (nlag in lags_array){
  #print(paste0(nlag," lags"))
  set.seed(120)
  
  #if (bootstrap_type=="region"){arlist=c(0,1,4)}else{arlist=c(0)}
  arlist <- c(0,1,4)
  for (ar in arlist){
    # AR models
    # up to AR(4), following Kahn et al. and Hsiang and Jina NBER
    print(paste0(nlag," lags, ar(",ar,")"))
    ## set up data
    
    # tx
    tx_coefs_dist <- data.frame("boot"=c(1:nboot))
    tx_interact_coefs_dist <- data.frame("boot"=c(1:nboot))
    # tmean
    tmean_coefs_dist <- data.frame("boot"=c(1:nboot))
    tmean_interact_coefs_dist <- data.frame("boot"=c(1:nboot))
    #tvar
    tvar_coefs_dist <- data.frame("boot"=c(1:nboot))
    tvar_interact_coefs_dist <- data.frame("boot"=c(1:nboot))
    
    for (ll in c(0:nlag)){
      tx_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
      tx_interact_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
      tmean_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
      tmean_interact_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
      tvar_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
      tvar_interact_coefs_dist[paste0("coef_lag",ll)] <- numeric(nboot)
    }
    
    # loop
    
    if (bootstrap==TRUE){
      for (n in c(1:nboot)){
        print(n)
        
        # bootstrap 
        if (bootstrap_type=="region"){
          ids <- unique(dat$region)
          boot_select <- sample(ids,size=length(ids),replace=T)
          df_boot <- sapply(boot_select, function(x) which(dat[,'region']==x))
        } else if (bootstrap_type=="country"){
          ids <- unique(dat$iso)
          boot_select <- sample(ids,size=length(ids),replace=T)
          df_boot <- sapply(boot_select, function(x) which(dat[,'iso']==x))
        } else if (bootstrap_type=="five_year"){
          ids <- unique(dat$year_block)
          boot_select <- sample(ids,size=length(ids),replace=T)
          df_boot <- sapply(boot_select, function(x) which(dat[,'year_block']==x))
        }
        data_boot <- dat[unlist(df_boot),]
        
        # build formula and run model
        form_dl <- paste0("growth ~ ",extr," + ",extr,"*",i)
        for (j in c(1:nlag)){
          form_dl <- paste0(form_dl," + ",extr,"_lag",j," + ",extr,"_lag",j,"*",i,"_lag",j)
        }
        form_dl <- paste0(form_dl," + ",trends,"t + t2 + p + var + var*seas")
        for (j in c(1:nlag)){form_dl <- paste0(form_dl," + t_lag",j," + t2_lag",j,
                                               " + p_lag",j," + var_lag",j,
                                               " + var_lag",j,"*seas")}
        
        # ar terms
        if (ar>0){
          for (aa in c(1:ar)){
            form_dl <- paste0(form_dl," + growth_lag",aa)
          }
        }
        
        # full formula
        form <- as.formula(paste0(form_dl," | ",fe," | 0 | ",cl))
        mdl <- felm(form,data=data_boot)
        
        # extract coefficients
        tx_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)[extr])
        tx_interact_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)[paste0(extr,":",i)])
        tmean_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)["t"])
        tmean_interact_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)["t2"])
        tvar_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)["var"])
        tvar_interact_coefs_dist[n,"coef_lag0"] <- as.numeric(coef(mdl)["var:seas"])
        for (ll in c(1:nlag)){
          tx_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(extr,"_lag",ll)])
          tx_interact_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0(extr,"_lag",ll,":",i,"_lag",ll)])
          tmean_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("t_lag",ll)])
          tmean_interact_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("t2_lag",ll)])
          tvar_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("var_lag",ll)])
          tvar_interact_coefs_dist[n,paste0("coef_lag",ll)] <- as.numeric(coef(mdl)[paste0("seas:var_lag",ll)])
        }
        
        #if (nlag>0){
        #  ss
        #}
      }
      
      if (ar==0){arlab=""}else if(ar>0){arlab=paste0("_ar",ar)}
      
      write.csv(tx_coefs_dist,paste0(loc_save_reg,extr,"_coefs_bootstrap_",bootstrap_type,"_lag",nlag,arlab,".csv"))
      write.csv(tx_interact_coefs_dist,paste0(loc_save_reg,extr,"_coefs_interact_bootstrap_",bootstrap_type,"_lag",nlag,arlab,".csv"))
      write.csv(tmean_coefs_dist,paste0(loc_save_reg,"temperature_coefs_",extr,"_model_bootstrap_",bootstrap_type,"_lag",nlag,arlab,".csv"))
      write.csv(tmean_interact_coefs_dist,paste0(loc_save_reg,"temperature_coefs_interact_",extr,"_model_bootstrap_",bootstrap_type,"_lag",nlag,arlab,".csv"))
      write.csv(tvar_coefs_dist,paste0(loc_save_reg,"variability_coefs_",extr,"_model_bootstrap_",bootstrap_type,"_lag",nlag,arlab,".csv"))
      write.csv(tvar_interact_coefs_dist,paste0(loc_save_reg,"variability_coefs_interact_",extr,"_model_bootstrap_",bootstrap_type,"_lag",nlag,arlab,".csv"))
      
    }
  }
}




################ 
#### Placebo test
################


# randomize within region and within year
# drop singleton rows from panel_rand or else
# R gets mad when we try to "sample"
panel %>% filter(t!=0) %>%
  group_by(region) %>% 
  filter(n()>1) -> panel_rand
nboot <- 1000
set.seed(120)
extr <- "tx5d"
i <- "t"
fe <- "region + time"
cl <- "0"
form <- as.formula(paste0("growth ~ t + t2 + tx_rand + tx_rand:",i," + var + var:seas + p | ",fe," | 0 | ",cl))


rand_coefs_nonrand <- data.frame("boot"=c(1:nboot),
                                 "coef_main"=c(1:nboot),
                                 "coef_interact"=c(1:nboot)) # original model
rand_coefs_overall <- data.frame("boot"=c(1:nboot),
                                 "coef_main"=c(1:nboot),
                                 "coef_interact"=c(1:nboot))
rand_coefs_withinregion <- data.frame("boot"=c(1:nboot),
                                      "coef_main"=c(1:nboot),
                                      "coef_interact"=c(1:nboot))
rand_coefs_withinyear <- data.frame("boot"=c(1:nboot),
                                    "coef_main"=c(1:nboot),
                                    "coef_interact"=c(1:nboot))

for (n in c(1:nboot)){
  print(n)
  for (k in c(1:4)){
    
    # randomize three different ways
    if (k==1){ # original model -- bootstrap by region
      panel_rand <- panel_rand %>% 
        mutate(tx_rand = (!!as.name(extr)))
      ids <- unique(panel_rand$region)
      regions_boot <- sample(ids,size=length(ids),replace=T)
      df_boot <- sapply(regions_boot, function(x) which(panel_rand[,'region']==x))
      panel_rand_final <- panel_rand[unlist(df_boot),]
    } else if (k==2){
      panel_rand_final <- panel_rand %>% 
        mutate(tx_rand = sample(!!as.name(extr)))
    } else if (k==3){
      panel_rand_final <- panel_rand %>% 
        group_by(time) %>%
        mutate(tx_rand = sample(!!as.name(extr)))
    } else {
      panel_rand_final <- panel_rand %>% 
        group_by(region) %>%
        mutate(tx_rand = sample(!!as.name(extr)))
    }
    
    # estimate model
    mdl <- felm(form,data=panel_rand_final)
    
    if (k==1){
      rand_coefs_nonrand[n,"coef_main"] <- coef(summary(mdl))["tx_rand","Estimate"]
      rand_coefs_nonrand[n,"coef_interact"] <- coef(summary(mdl))[paste0(i,":tx_rand"),"Estimate"]
    } else if (k==2){
      rand_coefs_overall[n,"coef_main"] <- coef(summary(mdl))["tx_rand","Estimate"]
      rand_coefs_overall[n,"coef_interact"] <- coef(summary(mdl))[paste0(i,":tx_rand"),"Estimate"]
    } else if (k==3){
      rand_coefs_withinyear[n,"coef_main"] <- coef(summary(mdl))["tx_rand","Estimate"]
      rand_coefs_withinyear[n,"coef_interact"] <- coef(summary(mdl))[paste0(i,":tx_rand"),"Estimate"]
    } else {
      rand_coefs_withinregion[n,"coef_main"] <- coef(summary(mdl))["tx_rand","Estimate"]
      rand_coefs_withinregion[n,"coef_interact"] <- coef(summary(mdl))[paste0(i,":tx_rand"),"Estimate"]
    }
  }
}
write.csv(rand_coefs_nonrand,paste0(loc_save_reg,extr,"_coefs_randomization_nonrandom.csv"))
write.csv(rand_coefs_overall,paste0(loc_save_reg,extr,"_coefs_randomization_fullsample.csv"))
write.csv(rand_coefs_withinyear,paste0(loc_save_reg,extr,"_coefs_randomization_withinyear.csv"))
write.csv(rand_coefs_withinregion,paste0(loc_save_reg,extr,"_coefs_randomization_withinregion.csv"))

print("done!")




################ 
#### Extreme degree days
################

panel %>% filter(t!=0) -> dat # when t is exactly 0 it's an error
dat$growth <- dat$growth*100
summary(felm(as.formula("growth ~ t + t2 + edd95 + edd95:t + var + var:seas + p | region + time | 0 | region"),
             data=dat))



# scaling
dat %>% group_by(region) %>%
  summarize(edd_sd = sd(edd95,na.rm=T)) -> sd_dat_edd
edd_scaling <- mean(sd_dat_edd$edd_sd,na.rm=T)
print(edd_scaling)

## bootstrap!
metrics <- c("edd95")
#metrics <- c("tx15d","tmonx")

nboot <- 1000
fe <- "region + time"
cl <- "0"
i <- "t" # interaction
panel %>% filter(t!=0,!is.na(growth)) -> dat

# bootstrap by region
for (mm in c(1:length(metrics))){
  extr <- metrics[mm]
  form <- as.formula(paste0("growth ~ t + t2 + ",extr," + ",extr,":",i," + var + var:seas + p | ",fe," | 0 | ",cl))
  print(extr)
  
  # set up data
  mdl_df <- data.frame("boot"=c(1:nboot),
                       "coef_main"=numeric(nboot),
                       "coef_interact"=numeric(nboot))
  
  # loop through bootstrap iterations
  set.seed(120)
  for (n in c(1:nboot)){
    print(n)
    
    # bootstrap by region
    ids <- unique(dat$region)
    regions_boot <- sample(ids,size=length(ids),replace=T)
    df_boot <- sapply(regions_boot, function(x) which(dat[,'region']==x))
    data_boot <- dat[unlist(df_boot),]
    
    # run model
    mdl <- felm(form,data=data_boot)
    
    # save all the different coefficients
    mdl_df[n,"coef_main"] <- coef(summary(mdl))[extr,"Estimate"]
    if (i=="t"){
      mdl_df[n,"coef_interact"] <- coef(summary(mdl))[paste0(i,":",extr),"Estimate"]
    } else {
      mdl_df[n,"coef_interact"] <- coef(summary(mdl))[paste0(extr,":",i),"Estimate"]
    }
    

  }
  # write out when done with iterations
  write.csv(mdl_df,paste0(loc_save_reg,extr,"_coefs_bootstrap_contemporaneous.csv"))
  print(paste0(loc_save_reg,extr,"_coefs_bootstrap_contemporaneous.csv"))
}