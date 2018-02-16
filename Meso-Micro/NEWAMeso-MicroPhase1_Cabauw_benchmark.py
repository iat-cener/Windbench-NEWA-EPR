#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:41:59 2018

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pandas as pd
import netCDF4 
from windrose import WindroseAxes
from scipy import interpolate
import scipy.integrate as integrate
import statsmodels.api as sm
import seaborn as sns
from scipy import stats

# Constants
g = 9.81    # [m s-2]
P0 = 100000 # Reference pressure [Pa]
T0 = 300    # Reference temperature for perturbation temperature [K]
kappa = 0.2854  # Poisson constant (R/Cp)
R_air = 287.058  # Specific gas constant for dry air [J kg-1 K-1]
Cp_air = 1005   # Specific heat of air [J kg-1 K-1]
omega = 7.2921159e-5    # angular speed of the Earth [rad/s]
K = 0.41    # von Karman constant

# Site 
siteID = 'Cabauw'
lat_s = 51.971   # degrees N
lon_s = 4.927    # degrees E
fc  = 2*omega*np.sin(lat_s*np.pi/180)     # Coriolis parameter [s-1]

# Evaluation period
datefrom = datetime.datetime(2006,1,1,0,0,0)
dateto = datetime.datetime(2006,12,30,23,50,0)
Hhub = 120.         # hub-height
Drot = 160.         # rotor diameter
zref = 80.          # Choose a reference height corresponding to a measurement level 

# Load observational data
dirobs = './'
fileobs = 'Cabauw_mast_20060101_20061230.nc'
ts = 60     
nodata = -999.0
f = netCDF4.Dataset(dirobs+fileobs, 'r')

time_obs = f.variables['time'][:]    
datetime_obs = netCDF4.num2date(time_obs,units='hours since 0001-01-01 00:00:00.0',calendar='gregorian')

ifrom_obs=0
for j in range(1,len(datetime_obs)):
    if datetime_obs[j] <= datefrom:
        ifrom_obs = j

ito_obs=0
for j in range(1,len(datetime_obs)):
    if datetime_obs[j] <= dateto:
        ito_obs = j+1


datetime_obs = datetime_obs[ifrom_obs:ito_obs]

# sesonr heights
zT_obs = f.variables['zT'][:]
zS_obs = f.variables['zS'][:]
zWD_obs = f.variables['zWD'][:]
zflux_obs = f.variables['zflux'][:]

# Timeseries of variables
WD_obs = pd.DataFrame(f.variables['WD'][ifrom_obs:ito_obs], index = datetime_obs, columns = zWD_obs)
S_obs = pd.DataFrame(f.variables['S'][ifrom_obs:ito_obs], index = datetime_obs, columns = zS_obs)
T_obs = pd.DataFrame(f.variables['T'][ifrom_obs:ito_obs], index = datetime_obs, columns = zT_obs)
us_obs = pd.DataFrame(f.variables['us'][ifrom_obs:ito_obs], index = datetime_obs, columns = zflux_obs)
wt_obs = pd.DataFrame(f.variables['wt'][ifrom_obs:ito_obs], index = datetime_obs, columns = zflux_obs)
zL_obs = zflux_obs/pd.DataFrame(f.variables['L'][ifrom_obs:ito_obs], index = datetime_obs, columns = zflux_obs)

# Load tendencies
dirtend = '.'
filetend = '/Cabauw_tendencies_w60_L9000.nc'
f = netCDF4.Dataset(dirtend+filetend, 'r')
ztend = f.variables['z'][:]
times = f.variables['time'][:]
idates = np.where(np.logical_and(times >= mdates.date2num(datefrom), 
                             times < mdates.date2num(dateto)))[0] 
Ug = pd.DataFrame(f.variables['Ug'][idates], index = mdates.num2date(f.variables['time'][idates]), columns = ztend)
Vg = pd.DataFrame(f.variables['Vg'][idates], index = mdates.num2date(f.variables['time'][idates]), columns = ztend)
Uadv = pd.DataFrame(f.variables['Uadv'][idates], index = mdates.num2date(f.variables['time'][idates]), columns = ztend)
Vadv = pd.DataFrame(f.variables['Vadv'][idates], index = mdates.num2date(f.variables['time'][idates]), columns = ztend)
Thadv = pd.DataFrame(f.variables['Thadv'][idates], index = mdates.num2date(f.variables['time'][idates]), columns = ztend)
Ug.index.tz = None; Vg.index.tz = None; Uadv.index.tz = None; Vadv.index.tz = None; Thadv.index.tz = None

# Load simulation data
dirsim = '.'
filesim = [dirsim + '/WRF-YSU.nc']
simID = ['WRF-YSU']
simtype = ['meso']
Nsim = len(filesim)

t = []; U = []; V = []; Th = []; z = []; S = []; WD = []; us = []
wt = []; T2 = []; TKE = []; zL = []
for isim in range(0,Nsim):
    f = netCDF4.Dataset(filesim[isim], 'r')
    times = f.variables['time'][:]
    idates = np.where(np.logical_and(times >= mdates.date2num(datefrom), 
                                 times < mdates.date2num(dateto)))[0] 
    z.append(f.variables['z'][:])
    zflux = f.variables['zflux'][:]
    t.append(pd.DataFrame(f.variables['time'][idates], index = mdates.num2date(f.variables['time'][idates])))       
    U.append(pd.DataFrame(f.variables['U'][idates,:], index = mdates.num2date(f.variables['time'][idates]), columns = f.variables['z'][:]))
    V.append(pd.DataFrame(f.variables['V'][idates,:], index = mdates.num2date(f.variables['time'][idates]), columns = f.variables['z'][:]))
    Th.append(pd.DataFrame(f.variables['Th'][idates,:], index = mdates.num2date(f.variables['time'][idates]), columns = f.variables['z'][:]))
    S.append((U[isim]**2 + V[isim]**2)**0.5)
    WD.append(180 + np.arctan2(U[isim],V[isim])*180/np.pi)
    us.append(pd.DataFrame(f.variables['ust'][idates], index = mdates.num2date(f.variables['time'][idates]), columns = zflux))
    wt.append(pd.DataFrame(f.variables['wt'][idates], index = mdates.num2date(f.variables['time'][idates]), columns = zflux))
    T2.append(pd.DataFrame(f.variables['T2'][idates], index = mdates.num2date(f.variables['time'][idates]), columns = zflux))
    zL.append(zflux/(-us[isim]**3/(K*(g/T2[isim])*wt[isim])))
    U[isim].index.tz = None; V[isim].index.tz = None; Th[isim].index.tz = None; S[isim].index.tz = None     # this is to avoid bug in pd.concat below
    WD[isim].index.tz = None; us[isim].index.tz = None; wt[isim].index.tz = None; T2[isim].index.tz = None
    zL[isim].index.tz = None
    f.close()
    
# Resample to hourly data
ts = 60
WD_obs = WD_obs.resample(str(ts)+'Min').mean().bfill()
S_obs = S_obs.resample(str(ts)+'Min').mean().bfill()
T_obs = T_obs.resample(str(ts)+'Min').mean().bfill()
us_obs = us_obs.resample(str(ts)+'Min').mean().bfill()
wt_obs = wt_obs.resample(str(ts)+'Min').mean().bfill()
zL_obs = zL_obs.resample(str(ts)+'Min').mean().bfill()
Ug = Ug.resample(str(ts)+'Min').mean().bfill()
Vg = Vg.resample(str(ts)+'Min').mean().bfill()
Uadv = Uadv.resample(str(ts)+'Min').mean().bfill()
Vadv = Vadv.resample(str(ts)+'Min').mean().bfill()
Thadv = Thadv.resample(str(ts)+'Min').mean().bfill()
Sadv = (Uadv**2 + Vadv**2)**0.5
WDadv = 180 + np.arctan2(U[isim],V[isim])*180/np.pi
Th_obs = T_obs + (g/Cp_air)*zT_obs
U_obs = S_obs*np.cos((270.-WD_obs)*np.pi/180)
V_obs = S_obs*np.sin((270.-WD_obs)*np.pi/180)

for isim in range(0,Nsim):
    U[isim] = U[isim].resample(str(ts)+'Min').mean().bfill()
    V[isim] = V[isim].resample(str(ts)+'Min').mean().bfill()
    WD[isim] = WD[isim].resample(str(ts)+'Min').mean().bfill()
    S[isim] = S[isim].resample(str(ts)+'Min').mean().bfill()
    Th[isim] = Th[isim].resample(str(ts)+'Min').mean().bfill()
    us[isim] = us[isim].resample(str(ts)+'Min').mean().bfill()
    wt[isim] = wt[isim].resample(str(ts)+'Min').mean().bfill()
    zL[isim] = zL[isim].resample(str(ts)+'Min').mean().bfill()


# interpolate to reference height
S_obs['zref'] = interpolate.interp1d(zS_obs, S_obs[S_obs.columns.values])(zref)
WD_obs['zref'] = interpolate.interp1d(zWD_obs, WD_obs[WD_obs.columns.values])(zref)
Th_obs['zref'] = interpolate.interp1d(zT_obs, Th_obs[Th_obs.columns.values])(zref)

for isim in range(0,Nsim):
    S[isim]['zref'] = interpolate.interp1d(z[isim], S[isim][S[isim].columns.values])(zref)
    WD[isim]['zref'] = interpolate.interp1d(z[isim], WD[isim][WD[isim].columns.values])(zref)
    Th[isim]['zref'] = interpolate.interp1d(z[isim], Th[isim][Th[isim].columns.values])(zref)

# Plot time series
datefromplot = datetime.datetime(2006,6,28,0,0,0)
datetoplot = datetime.datetime(2006,7,4,23,50,0)

Splot = S_obs['zref'][datefromplot:datetoplot].rename('Obs')
WDplot = WD_obs['zref'][datefromplot:datetoplot].rename('Obs')
Thplot = Th_obs['zref'][datefromplot:datetoplot].rename('Obs')
usplot = us_obs[datefromplot:datetoplot]
usplot.columns = ['Obs']
zLplot = zL_obs[datefromplot:datetoplot]
zLplot.columns = ['Obs']

for isim in range(0,Nsim):  # interpolate simulation data to zref
    subset = S[isim]['zref'][datefromplot:datetoplot].rename(simID[isim])    
    Splot = pd.concat([Splot,subset], axis = 1)
    subset = WD[isim]['zref'][datefromplot:datetoplot].rename(simID[isim])    
    WDplot = pd.concat([WDplot,subset], axis = 1)
    subset = Th[isim]['zref'][datefromplot:datetoplot].rename(simID[isim])    
    Thplot = pd.concat([Thplot,subset], axis = 1)
    subset = us[0][datefromplot:datetoplot]
    subset.columns = [simID[isim]]
    usplot = pd.concat([usplot,subset], axis = 1)
    subset = zL[0][datefromplot:datetoplot]
    subset.columns = [simID[isim]]
    zLplot = pd.concat([zLplot,subset], axis = 1)


figname = siteID + '_timeseries.png'
fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6),sharex=True, sharey=False)
Splot.plot(ax=axes[0,0], style=['.','-'], color =['silver','k'],legend=False, grid=True); axes[0,0].set_title('$S_{ref}$ [$m s^{-1}$], $z_{ref}$ = '+str(zref)+' m')
WDplot.plot(ax=axes[0,1], style=['.','-'], color =['silver','k'], grid=True); axes[0,1].set_title('$WD_{ref}$ ['+u'\N{DEGREE SIGN}'+']')
#usplot.plot(ax=axes[1,0], style=['.','-'], color =['silver','k'],legend=False, grid=True); axes[1,0].set_title('$u_*$ [$m s^{-1}$]')
zLplot.plot(ax=axes[1,0], style=['.','-'], color =['silver','k'],legend=False, grid=True, ylim = [-1,1]); axes[1,0].set_title('$z/L_0$')
Thplot.plot(ax=axes[1,1], style=['.','-'], color =['silver','k'],legend=False, grid=True); axes[1,1].set_title('$\Theta_{ref}$ [K]')
#plt.savefig(figname, dpi=300, bbox_inches='tight')

# Plot windorse
figname = siteID + '_windrose.png'
rose = pd.concat([S_obs[zref],WD_obs[zref]], axis = 1, keys = ['speed','direction']).dropna()
ax = WindroseAxes.from_ax()
ax.bar(rose.direction, rose.speed, normed=True, bins=np.arange(0.,25.,5.) , opening=0.8, edgecolor='black')
ax.set_legend()
#plt.savefig(figname, dpi=300, bbox_inches='tight')


# Compute quantities of interest
zrot = np.linspace(Hhub - 0.5*Drot, Hhub + 0.5*Drot, 1. + Drot/10, endpoint=True)

def rotor(z,Sz,WDz):
    # Rotor QoIs [m s-1]
    # z: heights [m] where the velocity and wind direction are known spanning the rotor diameter
    # Sz, WDz: Wind speed [m s-1] and direction [deg from N] at z levels [tdim,zdim]
    # Returns:
    #   REWS: rotor equivalent wind speed [m s-1]
    #   alpha: wind shear, power-law exponent from linear fit lg(U/Uhub) = alpha*log(z/zhub)
    #   alpha_R2: R-squared from least squares fit to linear function
    #   veer: wind veer, slope of linear function beta = WDz - WDhub = veer*(z - zhub)
    #   veer_R2: R-squared from least squares fit to linear function
    tdim = Sz.shape[0]
    zdim = Sz.shape[1]    
    Rrot = 0.5*(z[-1]-z[0])    
    Hhub = 0.5*(z[-1]+z[0])
    ihub = int(0.5*len(z))
    Arotor = np.pi*(Rrot)**2
    Uz = -Sz*np.sin(np.pi*WDz/180.)
    Vz = -Sz*np.cos(np.pi*WDz/180.)
    Shub = Sz[:,ihub]
    WDhub = WDz[:,ihub]       
    def cz(x,R,H):
        return 2.*(R**2 - (x-H)**2)**0.5
    sumA = np.zeros((Sz.shape[0]))    
    veer = np.zeros((Sz.shape[0]))    
    for i in range(0,zdim-1):
        Ai = integrate.quad(cz, z[i], z[i+1], args = (Rrot,Hhub))    
        Si = 0.5*(Sz[:,i+1]+Sz[:,i])
        Ui = 0.5*(Uz[:,i+1]+Uz[:,i])
        Vi = 0.5*(Vz[:,i+1]+Vz[:,i])   
        WDi = 180. + np.arctan2(Ui,Vi)*180./np.pi       
        betai = WDi - WDhub
        sumA = sumA + Ai[0]*(Si*np.cos(np.pi*betai/180.))**3
    
    REWS = (sumA/Arotor)**(1./3.)             
    
    alpha = np.zeros(tdim);    alpha_stderr = np.zeros(tdim); alpha_R2 = np.zeros(tdim)
    veer = np.zeros(tdim);     veer_stderr = np.zeros(tdim); veer_R2 = np.zeros(tdim)
    for it in range(0,tdim):
        regmodel = sm.OLS(np.log(Sz[it,:]/Shub[it]), np.log(z/Hhub))
        results = regmodel.fit()
        alpha[it] = results.params[0]
        alpha_stderr[it] = results.bse[0]
        alpha_R2[it] = results.rsquared
        regmodel = sm.OLS(WDz[it,:] - WDhub[it], z - Hhub)
        results = regmodel.fit()
        veer[it] = results.params[0]
        veer_stderr[it] = results.bse[0]
        veer_R2[it] = results.rsquared
    
    return REWS, Shub, WDhub, alpha, alpha_R2, veer, veer_R2

Srews_obs = interpolate.interp1d(zS_obs,S_obs[zS_obs].values)(zrot)
Urews_obs = interpolate.interp1d(zS_obs,U_obs[zS_obs].values)(zrot)
Vrews_obs = interpolate.interp1d(zS_obs,V_obs[zS_obs].values)(zrot)
WDrews_obs = 180. + np.arctan2(Urews_obs,Vrews_obs)*180./np.pi
REWS_obs, Shub_obs, WDhub_obs, alpha_obs, alpha_R2_obs, veer_obs, veer_R2_obs = rotor(zrot,Srews_obs,WDrews_obs)

Rotor_obs = pd.DataFrame(REWS_obs, index = S_obs.index, columns = {'REWS'})
Rotor_obs['Shub'] = pd.DataFrame(Shub_obs, index = S_obs.index)
Rotor_obs['WDhub'] = pd.DataFrame(WDhub_obs, index = S_obs.index)
Rotor_obs['alpha'] = pd.DataFrame(alpha_obs, index = S_obs.index)
Rotor_obs['veer'] = pd.DataFrame(veer_obs, index = S_obs.index)
Rotor_obs['Sref'] = S_obs['zref']
Rotor_obs['WDref'] = WD_obs['zref']
Rotor_obs['zL0'] = zL_obs

Rotor = []
for isim in range(0,Nsim):
    Srews_sim = interpolate.interp1d(z[isim],S[isim][z[isim]].values)(zrot)
    Urews_sim = interpolate.interp1d(z[isim],U[isim][z[isim]].values)(zrot)
    Vrews_sim = interpolate.interp1d(z[isim],V[isim][z[isim]].values)(zrot)
    WDrews_sim = 180. + np.arctan2(Urews_sim,Vrews_sim)*180./np.pi       
    REWS0, Shub0, WDhub0, alpha0, alpha_R20, veer0, veer_R20 = rotor(zrot,Srews_sim,WDrews_sim)
    
    Rotor0 = pd.DataFrame(REWS0, index = S[isim].index, columns = {'REWS'})
    Rotor0['Shub'] = pd.DataFrame(Shub0, index = S[isim].index)    
    Rotor0['WDhub'] = pd.DataFrame(WDhub0, index = S[isim].index)    
    Rotor0['alpha'] = pd.DataFrame(alpha0, index = S[isim].index)    
    Rotor0['veer'] = pd.DataFrame(veer0, index = S[isim].index)
    Rotor0['Sref'] = S[isim]['zref']
    Rotor0['WDref'] = WD[isim]['zref']
    Rotor0['zL0'] = zL[isim]    
    
    Rotor.append(Rotor0)

# Compute distributions
Sbins = np.hstack((np.arange(0,16,1)))
Sbins_label = Sbins[0:-1]+1
WDbins = np.arange(-11.25,360.+11.25,22.5)
WDbins_label = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                'S','SSW','SW','WSW','W','WNW','NW','NNW']
zLbins = [-20,-2, -0.6, -0.2, -0.02, 0.02, 0.2, 0.6, 2, 20]
zLbins_label = ['xu','vu','u','wu','n','ws','s','vs','xs']

NzL = len(zLbins_label)
NWD = len(WDbins_label)
NS = len(Sbins_label)

bins = [Sbins,zLbins]
x =Rotor_obs['Sref'].values
y = Rotor_obs['zL0'].values

values = Rotor_obs['REWS'].values
statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values, statistic='count', bins=bins, expand_binnumbers = True)
N_SzL_obs = pd.DataFrame(statistic, index=Sbins_label, columns=zLbins_label)
N_S_obs = np.sum(N_SzL_obs, axis = 1).rename('pdf_obs')

bins = [WDbins,zLbins]
x =Rotor_obs['WDref'].values
x[x>WDbins[-1]] = x[x>WDbins[-1]]-360
y = Rotor_obs['zL0'].values

values = Rotor_obs['REWS'].values
statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values, statistic='count', bins=bins, expand_binnumbers = True)
N_WDzL_obs = pd.DataFrame(statistic, index=WDbins_label, columns=zLbins_label)
N_zL_obs = np.sum(N_WDzL_obs, axis = 0).rename('pdf_obs')
N_WD_obs = np.sum(N_WDzL_obs, axis = 1).rename('pdf_obs')

statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values, statistic='mean', bins=bins)
REWS_WDzL_obs = pd.DataFrame(statistic, index=WDbins_label, columns=zLbins_label)

values = Rotor_obs['alpha'].values
statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values, statistic='mean', bins=bins)
alpha_WDzL_obs = pd.DataFrame(statistic, index=WDbins_label, columns=zLbins_label)

values = Rotor_obs['veer'].values
statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values, statistic='mean', bins=bins)
veer_WDzL_obs = pd.DataFrame(statistic, index=WDbins_label, columns=zLbins_label)

N_SzL = []; N_S = []
N_WDzL = []; N_WD = []; N_zL = []; 
REWS_WDzL = []; alpha_WDzL = []; veer_WDzL = []

for isim in range(0,Nsim):
    x = Rotor[isim]['Sref'].values
    y = Rotor[isim]['zL0'].values
    bins = [Sbins,zLbins]
    
    values = Rotor[isim]['REWS'].values
    statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values, statistic='count', bins=bins, expand_binnumbers = True)
    N_SzL.append(pd.DataFrame(statistic, index=Sbins_label, columns=zLbins_label))
    N_S.append(np.sum(N_SzL[isim], axis = 1).rename('pdf_sim'))
    
    x = Rotor[isim]['WDref'].values
    x[x>WDbins[-1]] = x[x>WDbins[-1]]-360
    y = Rotor[isim]['zL0'].values
    bins = [WDbins,zLbins]
    
    values = Rotor[isim]['REWS'].values
    statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values, statistic='count', bins=bins, expand_binnumbers = True)
    N_WDzL.append(pd.DataFrame(statistic, index=WDbins_label, columns=zLbins_label))
    N_zL.append(np.sum(N_WDzL[isim], axis = 0).rename('pdf_sim'))
    N_WD.append(np.sum(N_WDzL[isim], axis = 1).rename('pdf_sim'))
    
    statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values, statistic='mean', bins=bins)
    REWS_WDzL.append(pd.DataFrame(statistic, index=WDbins_label, columns=zLbins_label))
    
    values = Rotor[isim]['alpha'].values
    statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values, statistic='mean', bins=bins)
    alpha_WDzL.append(pd.DataFrame(statistic, index=WDbins_label, columns=zLbins_label))
    
    values = Rotor[isim]['veer'].values
    statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values, statistic='mean', bins=bins)
    veer_WDzL.append(pd.DataFrame(statistic, index=WDbins_label, columns=zLbins_label))
    
# Compute bin-based bias 
Bias_REWS = []; Bias_alpha = []; Bias_veer = []
for isim in range(0,Nsim):
    Bias_REWS.append(REWS_WDzL_obs - REWS_WDzL[isim])    
    Bias_alpha.append(alpha_WDzL_obs - alpha_WDzL[isim])    
    Bias_veer.append(veer_WDzL_obs - veer_WDzL[isim])
    
# Plot normalized pdf f(WD,zL) & f(S,zL) 
Nnorm_WDzL_obs = N_WDzL_obs.div(N_WD_obs, axis=0)
Nnorm_WDzL = []
for isim in range(0,Nsim):
    Nnorm_WDzL.append(N_WDzL[isim].div(N_WD[isim], axis=0))

figname = siteID + '_zLvsWD_dist.png'
fig, axes = plt.subplots(1,2, figsize=(12,4))
cmap = plt.get_cmap('bwr')
zLcolors = np.flipud(cmap(np.linspace(0.,NzL,NzL)/NzL))
ax1=Nnorm_WDzL_obs.plot.bar(ax=axes[0], stacked=True, color=zLcolors, align='center', width=1.0, legend=False, rot=90, use_index = False)
ax2=(N_WD_obs/N_WD_obs.sum()).plot(ax=axes[0], secondary_y=True, style='grey',legend=False, rot=90, use_index = False)
ax2.set_xticklabels(WDbins_label)
ax1.set_title('Obs')
ax1.set_ylabel('$pdf_{norm}$($z/L_0$)')
ax2.set_ylabel('$pdf$($WD_{ref}$), $z_{ref}$ = '+str(zref)+' m', rotation=-90, labelpad=15)
ax1.set_yticks(np.linspace(0,1.,6))
ax1.set_ylim([0,1.])
ax2.set_yticks(np.linspace(0,0.2,6))

ax3=Nnorm_WDzL[isim].plot.bar(ax=axes[1], stacked=True, color=zLcolors, align='center', width=1.0,legend=False, rot=90, use_index = False)
ax4=(N_WD_obs/N_WD_obs.sum()).plot(ax=axes[1], secondary_y=True, style='grey',legend=False, rot=90, use_index = False)
ax4=(N_WD[isim]/N_WD[isim].sum()).plot(ax=axes[1], secondary_y=True, style='k',legend=False, rot=90, use_index = False)
ax4.set_xticklabels(WDbins_label)
ax3.set_title(simID[isim])
ax3.set_ylabel('$pdf_{norm}$($z/L_0$)')
ax4.set_ylabel('$pdf$($WD_{ref}$), $z_{ref}$ = '+str(zref)+' m', rotation=-90, labelpad=15)
ax3.set_yticks(np.linspace(0,1.,6))
ax3.set_ylim([0,1.])
ax4.set_yticks(np.linspace(0,0.2,6))
h1, l1 = ax3.get_legend_handles_labels()
h2, l2 = ax4.get_legend_handles_labels()
plt.legend(h1+h2, l1+l2, bbox_to_anchor=(1.4, 1))
plt.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.2)
   
Nnorm_SzL_obs = N_SzL_obs.div(N_S_obs, axis=0)
Nnorm_SzL = []
for isim in range(0,Nsim):
    Nnorm_SzL.append(N_SzL[isim].div(N_S[isim], axis=0))

fig, axes = plt.subplots(1,2, figsize=(12,4))
ax5=Nnorm_SzL_obs.plot.bar(ax=axes[0], stacked=True, color=zLcolors, align='center', width=1.0, legend=False, rot=90, use_index = False)
ax6=(N_S_obs/N_S_obs.sum()).plot(ax=axes[0], secondary_y=True, style='grey',legend=False, rot=90, use_index = False) 
ax5.set_xticklabels(Sbins_label)
ax5.set_title('Obs')
ax5.set_ylabel('$pdf_{norm}$($z/L_0$)')
ax6.set_ylabel('$pdf$($S_{ref}$), $z_{ref}$ = '+str(zref)+' m', rotation=-90, labelpad=15)
ax5.set_yticks(np.linspace(0,1.,6))
ax5.set_ylim([0,1.])
ax6.set_yticks(np.linspace(0,0.2,6))

ax7=Nnorm_SzL[isim].plot.bar(ax=axes[1], stacked=True, color=zLcolors, align='center', width=1.0,legend=False, rot=90, use_index = False)
ax8=(N_S_obs/N_S_obs.sum()).plot(ax=axes[1], secondary_y=True, style='grey',legend=False, rot=90, use_index = False)
ax8=(N_S[isim]/N_S[isim].sum()).plot(ax=axes[1], secondary_y=True, style='k',legend=False, rot=90, use_index = False)
ax8.set_xticklabels(Sbins_label)
ax7.set_title(simID[isim])
ax7.set_ylabel('$pdf_{norm}$($z/L_0$)')
ax8.set_ylabel('$pdf$($S_{ref}$), $z_{ref}$ = '+str(zref)+' m', rotation=-90, labelpad=15)
ax7.set_yticks(np.linspace(0,1.,6))
ax7.set_ylim([0,1.])
ax8.set_yticks(np.linspace(0,0.2,6))
h1, l1 = ax7.get_legend_handles_labels()
h2, l2 = ax8.get_legend_handles_labels()
plt.legend(h1+h2, l1+l2, bbox_to_anchor=(1.4, 1))
plt.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.2)    

# Plot QoI and associated bias vs WD-zL bins   
isim = 0

Z = [REWS_WDzL_obs, REWS_WDzL[isim], Bias_REWS[isim],
         alpha_WDzL_obs, alpha_WDzL[isim], Bias_alpha[isim],
         veer_WDzL_obs, veer_WDzL[isim], Bias_veer[isim]]

Ztitle = ['Obs', simID[isim], 'Bias = Obs - Sim']
Zlabel = ('$REWS$ [$m s^{-1}$]','Wind shear ('+r'$\alpha$'+')','Wind veer ('+r'$\beta$'+')')
Xlabel = '$WD_{ref}$'
Ylabel =  '$z/L_0$'

Zlevels = [np.linspace(2,14,13), np.linspace(2,14,13), np.linspace(-5,5,11),
           np.linspace(-0.1,0.4,11), np.linspace(-0.1,0.4,11), np.linspace(-0.5,0.5,11),
           np.linspace(-0.5,0.5,11), np.linspace(-0.5,0.5,11), np.linspace(-0.5,0.5,11)]    
Zcmap = [plt.get_cmap('jet'),plt.get_cmap('jet'),plt.get_cmap('bwr')]

figname = siteID+'_'+simID[isim]+'_QoImap.png'
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(14,10))
CF = []
for iax in range (0,9):
    ix,iy = np.unravel_index(iax,(3,3))
    sns.heatmap(Z[iax].T, ax = ax[ix,iy], cmap=Zcmap[iy], 
                vmin=Zlevels[iax].min(), vmax=Zlevels[iax].max(),
                cbar_kws={'boundaries':Zlevels[iax]}, linewidths=.1)
    if iy == 1:
        ax[ix,iy].contour(N_WDzL[isim].values.T, linewidths = 1, cmap = plt.get_cmap('copper'))
    else:
        ax[ix,iy].contour(N_WDzL_obs.values.T, linewidths = 1, cmap = plt.get_cmap('copper'))
        
#    sns.kdeplot(Rotor_obs['WDref'].values, Rotor_obs['zL0'].values), 
#                               n_levels=5, cmap="Purples_d", cbar=True)
    
    #CS = ax[ix,iy].contour(X,Y,WDH_obs.T, linewidths=0.5, colors='k')
    ax[ix,iy].set_facecolor('grey')
    ax[ix,iy].set_title(Ztitle[iy]+': '+Zlabel[ix])
plt.tight_layout()

# Mean vertical profiles vs stability for a given wind direction sector 
sector = {'SW': [213.75, 236.25],  # most frequent
          'ESE': [101.25, 123.75], # GABLS3
          'NW': [303.75, 326.25]}  # coastal

Sbin = [4.,25.]         # filter relevant velocity range 
sector = sector['ESE']
WDcenter = 0.5*(sector[0]+sector[1])

# We choose the observed WD and stability to classify bins to make sure we are
# choosing the same timestamps for all the simulations

Uadv['WD0'] = WD_obs['zref']; Uadv['zL0'] = zL_obs; Uadv['S0'] = S_obs['zref']
Uadv_sector = Uadv.loc[lambda df: (df.WD0 >= sector[0]) & (df.WD0 <  sector[1]) &
                                  (df.S0 >= Sbin[0]) & (df.S0 <= Sbin[1]),:]
Vadv['WD0'] = WD_obs['zref']; Vadv['zL0'] = zL_obs; Vadv['S0'] = S_obs['zref']
Vadv_sector = Vadv.loc[lambda df: (df.WD0 >= sector[0]) & (df.WD0 <  sector[1]) &
                                  (df.S0 >= Sbin[0]) & (df.S0 <= Sbin[1]),:]
Thadv['WD0'] = WD_obs['zref']; Thadv['zL0'] = zL_obs; Thadv['S0'] = S_obs['zref']
Thadv_sector = Thadv.loc[lambda df: (df.WD0 >= sector[0]) & (df.WD0 <  sector[1]) &
                                  (df.S0 >= Sbin[0]) & (df.S0 <= Sbin[1]),:]
Ug['WD0'] = WD_obs['zref']; Ug['zL0'] = zL_obs; Ug['S0'] = S_obs['zref']
Ug_sector = Ug.loc[lambda df: (df.WD0 >= sector[0]) & (df.WD0 <  sector[1]) &
                                  (df.S0 >= Sbin[0]) & (df.S0 <= Sbin[1]),:]
Vg['WD0'] = WD_obs['zref']; Vg['zL0'] = zL_obs; Vg['S0'] = S_obs['zref']
Vg_sector = Vg.loc[lambda df: (df.WD0 >= sector[0]) & (df.WD0 <  sector[1]) &
                                  (df.S0 >= Sbin[0]) & (df.S0 <= Sbin[1]),:]
S_obs['WD0'] = WD_obs['zref']; S_obs['zL0'] = zL_obs; S_obs['S0'] = S_obs['zref']
S_obs_sector = S_obs.loc[lambda df: (df.WD0 >= sector[0]) & (df.WD0 <  sector[1]) &
                                  (df.S0 >= Sbin[0]) & (df.S0 <= Sbin[1]),:]
WD_obs['WD0'] = WD_obs['zref']; WD_obs['zL0'] = zL_obs; WD_obs['S0'] = S_obs['zref']
WD_obs_sector = WD_obs.loc[lambda df: (df.WD0 >= sector[0]) & (df.WD0 <  sector[1]) &
                                  (df.S0 >= Sbin[0]) & (df.S0 <= Sbin[1]),:]
Th_obs['WD0'] = WD_obs['zref']; Th_obs['zL0'] = zL_obs; Th_obs['S0'] = S_obs['zref']
Th_obs_sector = Th_obs.loc[lambda df: (df.WD0 >= sector[0]) & (df.WD0 <  sector[1]) &
                                  (df.S0 >= Sbin[0]) & (df.S0 <= Sbin[1]),:]

S_sector = []; WD_sector = []; U_sector = []; V_sector = []; Th_sector = []
for isim in range(0,Nsim):    
    U[isim]['WD0'] = WD_obs['zref']; U[isim]['zL0'] = zL_obs; U[isim]['S0'] = S_obs['zref']
    U_sector.append(U[isim].loc[lambda df: (df.WD0 >= sector[0]) & (df.WD0 <  sector[1]) &
                                           (df.S0 >= Sbin[0]) & (df.S0 <= Sbin[1]),:])
    V[isim]['WD0'] = WD_obs['zref']; V[isim]['zL0'] = zL_obs; V[isim]['S0'] = S_obs['zref']
    V_sector.append(V[isim].loc[lambda df: (df.WD0 >= sector[0]) & (df.WD0 <  sector[1]) &
                                           (df.S0 >= Sbin[0]) & (df.S0 <= Sbin[1]),:])
    Th[isim]['WD0'] = WD_obs['zref']; Th[isim]['zL0'] = zL_obs; Th[isim]['S0'] = S_obs['zref']
    Th_sector.append(Th[isim].loc[lambda df: (df.WD0 >= sector[0]) & (df.WD0 <  sector[1]) &
                                           (df.S0 >= Sbin[0]) & (df.S0 <= Sbin[1]),:])
    S[isim]['WD0'] = WD_obs['zref']; S[isim]['zL0'] = zL_obs; S[isim]['S0'] = S_obs['zref']
    S_sector.append(S[isim].loc[lambda df: (df.WD0 >= sector[0]) & (df.WD0 <  sector[1]) &
                                           (df.S0 >= Sbin[0]) & (df.S0 <= Sbin[1]),:])
    WD[isim]['WD0'] = WD_obs['zref']; WD[isim]['zL0'] = zL_obs; WD[isim]['S0'] = S_obs['zref']
    WD_sector.append(WD[isim].loc[lambda df: (df.WD0 >= sector[0]) & (df.WD0 <  sector[1]) &
                                           (df.S0 >= Sbin[0]) & (df.S0 <= Sbin[1]),:])    
    
# Compute mean profiles for each stability class 

Uadv_prof = []; Vadv_prof = []; Thadv_prof = [] 
Ug_prof = []; Vg_prof = []
Sadv_prof = []; WDadv_prof = []
Sg_prof = []; WDg_prof = []
S_obs_prof = []; WD_obs_prof = []; Th_obs_prof = []
Sgadv_prof = []; WDgadv_prof = []
NzL_obs = []

for izL in range(0,NzL):
    Uadv_sector_zL = Uadv_sector.loc[(lambda df: (df.zL0 >= zLbins[izL]) &
                                                (df.zL0 < zLbins[izL+1])),:].values
    Uadv_prof.append(np.nanmean(Uadv_sector_zL, axis = 0))
    Vadv_sector_zL = Vadv_sector.loc[(lambda df: (df.zL0 >= zLbins[izL]) &
                                                (df.zL0 < zLbins[izL+1])),:].values
    Vadv_prof.append(np.nanmean(Vadv_sector_zL, axis = 0))
    Thadv_sector_zL = Thadv_sector.loc[(lambda df: (df.zL0 >= zLbins[izL]) &
                                                (df.zL0 < zLbins[izL+1])),:].values
    Thadv_prof.append(np.nanmean(Thadv_sector_zL, axis = 0))
    Ug_sector_zL = Ug_sector.loc[(lambda df: (df.zL0 >= zLbins[izL]) &
                                             (df.zL0 < zLbins[izL+1])),:].values
    Ug_prof.append(np.nanmean(Ug_sector_zL, axis = 0))
    Vg_sector_zL = Vg_sector.loc[(lambda df: (df.zL0 >= zLbins[izL]) &
                                             (df.zL0 < zLbins[izL+1])),:].values
    Vg_prof.append(np.nanmean(Vg_sector_zL, axis = 0))
    
    Sadv_prof.append((Uadv_prof[izL]**2 + Vadv_prof[izL]**2)**0.5)
    WDadv_prof.append(180. + np.arctan2(Uadv_prof[izL],Vadv_prof[izL])*180./np.pi)
    Sg_prof.append((Ug_prof[izL]**2 + Vg_prof[izL]**2)**0.5)
    WDg_prof.append(180. + np.arctan2(-Vg_prof[izL],Ug_prof[izL])*180./np.pi)
    
    S_obs_sector_zL = S_obs_sector.loc[(lambda df: (df.zL0 >= zLbins[izL]) &
                                                   (df.zL0 < zLbins[izL+1])),:].values
    S_obs_prof.append(np.nanmean(S_obs_sector_zL, axis = 0))
    WD_obs_sector_zL = WD_obs_sector.loc[(lambda df: (df.zL0 >= zLbins[izL]) &
                                                   (df.zL0 < zLbins[izL+1])),:].values
    WD_obs_prof.append(np.nanmean(WD_obs_sector_zL, axis = 0))
    Th_obs_sector_zL = Th_obs_sector.loc[(lambda df: (df.zL0 >= zLbins[izL]) &
                                                   (df.zL0 < zLbins[izL+1])),:].values
    Th_obs_prof.append(np.nanmean(Th_obs_sector_zL, axis = 0))
    
    NzL_obs.append(S_obs_sector_zL.shape[0])
    
    Sgadv_prof.append(((-Vg_prof[izL]+Uadv_prof[izL])**2 + (Ug_prof[izL]+Vadv_prof[izL])**2)**0.5)
    WDgadv_prof.append(180 + np.arctan2(-Vg_prof[izL]+Uadv_prof[izL],Ug_prof[izL]+Vadv_prof[izL])*180/np.pi)    


U_sector_zL = []; V_sector_zL = []; Th_sector_zL = []; 
U_prof = []; V_prof = []; Th_prof = []; S_prof = []; WD_prof = []
for isim in range(0,Nsim):
    U_profsim = []; V_profsim = []; Th_profsim = []; S_profsim = []; WD_profsim = []
    for izL in range(0,NzL):
        U_sector_zL = U_sector[isim].loc[(lambda df: (df.zL0 >= zLbins[izL]) &
                                               (df.zL0 < zLbins[izL+1])),:].values
        U_profsim.append(np.nanmean(U_sector_zL, axis = 0))
        V_sector_zL = V_sector[isim].loc[(lambda df: (df.zL0 >= zLbins[izL]) &
                                               (df.zL0 < zLbins[izL+1])),:].values
        V_profsim.append(np.nanmean(V_sector_zL, axis = 0))
        Th_sector_zL = Th_sector[isim].loc[(lambda df: (df.zL0 >= zLbins[izL]) &
                                               (df.zL0 < zLbins[izL+1])),:].values
        Th_profsim.append(np.nanmean(Th_sector_zL, axis = 0))    
        S_profsim.append((U_profsim[izL]**2 + V_profsim[izL]**2)**0.5)
        WD_profsim.append(180. + np.arctan2(U_profsim[izL],V_profsim[izL])*180./np.pi)
    U_prof.append(U_profsim); V_prof.append(V_profsim); Th_prof.append(Th_profsim)
    S_prof.append(S_profsim); WD_prof.append(WD_profsim)
    
# Plot vertical profiles for different stabilities at a given WD sector
zLplot=[3,4,5,6,7] 
NzLplot = len(zLplot)

ZS = []; zz = []
ZSobs = []; zzobs = []
ZSg = []
iz = np.where(abs(z[isim]-zref)==min(abs(z[isim]-zref)))[0][0]
Nz = len(z[isim])
for izL0 in range(0,NzLplot):
    izL = zLplot[izL0]
    Sref = interpolate.interp1d(zS_obs,S_obs_prof[izL][0:-4])(zref)
    WDref = interpolate.interp1d(zWD_obs,WD_obs_prof[izL][0:-4])(zref)
    Thref = interpolate.interp1d(zT_obs,Th_obs_prof[izL][0:-4])(zref)
    Sgref = interpolate.interp1d(ztend,Sg_prof[izL][0:-3])(zref)
    WDgref = interpolate.interp1d(ztend,WDg_prof[izL][0:-3])(zref)
    ZSobs.append((S_obs_prof[izL][0:-4], WD_obs_prof[izL][0:-4]-WDref, Th_obs_prof[izL][0:-4]-Thref))
    zzobs.append((zS_obs, zWD_obs, zT_obs))
    ZSg.append((Sg_prof[izL][0:-3], WDg_prof[izL][0:-3]-WDgref))
    for isim in range(0,Nsim):    
        Sref = interpolate.interp1d(z[isim],S_prof[isim][izL][0:-3])(zref)
        WDref = interpolate.interp1d(z[isim],WD_prof[isim][izL][0:-3])(zref)
        Thref = interpolate.interp1d(z[isim],Th_prof[isim][izL][0:-4])(zref)
        ZS.append((S_prof[isim][izL][0:-3], WD_prof[isim][izL][0:-3]-WDref, Th_prof[isim][izL][0:-4]-Thref))
        zz.append((z[isim],z[isim],z[isim]))


ZSlabel = (('$S$', '$(WD-WD_{ref})$ ['+u'\N{DEGREE SIGN}'+']', '$\Theta-\Theta_{zref}$ ['+u'\N{DEGREE SIGN}'+'C]'))
ZSlim =  [[0, 20],[-20., 20],[-3, 3]]

linestylesim = ['k-','b.-','r.-','c.-','m-','g-','y-','c-','b-.','r--']
linestylesim = ['k-','b-','r-','c-','m-','g.-','y.-','c.-','b-.','r--'] # line style for each simulation
lwidth = np.array([2,1,1,1,1,1,1,1,2,2])
linewidthsim = np.array([2,1,1,1,1,1,1,1,2,2])

figname = siteID+"%.0f"%(WDcenter)+'_SWDThprof.png'
zLcolors[4] = np.array([0,0,0,1]) # neutral color = black

fig,ax = plt.subplots(nrows=len(zLplot), ncols=3, sharex='col', sharey='row', figsize=(8,8))
Nticks = 5
yrotor1 = np.array([Hhub - 0.5*Drot, Hhub - 0.5*Drot])
yrotor2 = np.array([Hhub + 0.5*Drot, Hhub + 0.5*Drot])
zlim = 300.0
for iax in range (0,NzLplot*3):
    ix,iy = np.unravel_index(iax,(NzLplot,3))
    ax[ix,iy].plot(ZSobs[ix][iy],zzobs[ix][iy],'ok', color = 'silver', label = 'Obs')
    if iy == 0:
        ax[ix,iy].plot(ZSg[ix][iy], ztend, '-.g', label = 'Geos.Wind', linewidth = 2)
        ax[ix,iy].set_title(zLbins_label[zLplot[ix]]+' (N =' + str(NzL_obs[zLplot[ix]])+')', loc = 'left')
    if iy == 1:
        ax[ix,iy].plot(ZSg[ix][iy], ztend, '-.g', label = 'Geos.Wind', linewidth = 2)
    for isim in range(0,Nsim):
        ax[ix,iy].plot(ZS[ix][iy][1:Nz], zz[ix][iy][1:],
          linestylesim[isim], linewidth = linewidthsim[isim], label = simID[isim])
    
    ax[ix,iy].grid(which='major',color='k',linestyle=':')
    ax[ix,iy].plot(ZSlim[iy],yrotor1,'--',color = 'grey')
    ax[ix,iy].plot(ZSlim[iy],yrotor2,'--',color = 'grey')
    ax[ix,iy].set_ylim([0, zlim])
    ax[ix,iy].set_xlim(ZSlim[iy])
    if iy == 0:
        ax[ix,iy].set_ylabel('$z$ [$m$]')
    if ix == NzLplot-1:
        ax[ix,iy].set_xlabel(ZSlabel[iy])
    if iax == 1:
        ax[ix,iy].set_title('$WD_{ref}$ = ['+"%.2f"%(sector[0])+', '+"%.2f"%(sector[1])+']'+u'\N{DEGREE SIGN}') 
    if iax == 0:
        ax[ix,iy].legend(bbox_to_anchor=(4.1, 1.05))

plt.tight_layout()

