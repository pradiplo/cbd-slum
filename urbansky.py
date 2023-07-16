#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:54:47 2023

@author: kunogenta
"""
import matplotlib
import matplotlib.pyplot as plt
from datetime import date
import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit
import scipy
import pickle
import gzip
import shapely
import random

today = date.today().strftime("%d_%m_%Y")

### World Gov stats#####
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#botanical_continents=gpd.read_file("data/level1.geojson")#https://github.com/tdwg/wgsrpd/blob/master/geojson/level1.geojson
regions="./data/world_gov/UNSD — Methodology.csv"#https://unstats.un.org/unsd/methodology/m49/overview/
#gdp="./data/world_gov/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5607117.csv" #https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
#pop="./data/world_gov/API_SP.POP.TOTL_DS2_en_csv_v2_5607187.csv"#https://data.worldbank.org/indicator/SP.POP.TOTL
#https://datacatalog.worldbank.org/search/dataset/0038272/World-Bank-Official-Boundaries
gdpop="./data/world_gov/WB_countries_Admin0_lowres.geojson"
co2="./data/world_gov/API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5607498.csv" #https://data.worldbank.org/indicator/EN.ATM.CO2E.PC?view=map
slumwb="./data/world_gov/API_EN.POP.SLUM.UR.ZS_DS2_en_csv_v2_5609682.csv"
world_gov=pd.read_csv(regions,error_bad_lines=False, sep=";")

world_gov=world_gov.set_index("ISO-alpha3 Code")
world_gov=world_gov[["Sub-region Name", "Region Name", 
           "Least Developed Countries (LDC)",                                  
           "Land Locked Developing Countries (LLDC)",                           
           "Small Island Developing States (SIDS)"]
          ]
world_gov.loc["TWN", "Region Name"]="Asia"
world_gov.loc["TWN", "Sub-region Name"]="Eastern Asia"
world_gov.loc["XKO", "Region Name"]="Europe"
world_gov.loc["XNC", "Region Name"]="Europe"
dev=world_gov[["Least Developed Countries (LDC)",                     
"Land Locked Developing Countries (LLDC)",                           
"Small Island Developing States (SIDS)"]].replace("x", 1).fillna(0).sum(1)
world_gov["developing"]=dev>0
world_gov["developing"]="developing:"+world_gov["developing"].astype(str)
world_gov["Region Name"]=world_gov["Region Name"].replace({"Oceania":"Asia/Oceania", "Asia":"Asia/Oceania"})

bank=gpd.read_file(gdpop)

bank=bank[[  'ISO_A3','POP_EST', 'POP_RANK', 'GDP_MD_EST', 'POP_YEAR',
'LASTCENSUS', 'GDP_YEAR', 'ECONOMY', 'INCOME_GRP', 'CONTINENT',
'REGION_UN', 'SUBREGION', 'REGION_WB', "geometry"]]
world_gov=bank.set_index("ISO_A3").join(world_gov[["developing", "Region Name"]])
world_gov["developed"]="developed:"+world_gov["ECONOMY"].str.contains("Developed").astype(str).str.replace("nan", "False")
world_gov["high_income"]="high income:"+world_gov["INCOME_GRP"].str.contains("High income").astype(str).str.replace("nan", "False")

co2=pd.read_csv(co2, skiprows=3)
numcol=co2.select_dtypes(include=numerics).columns
for i, dx in co2.set_index("Country Code").iterrows():
    
    recent=dx[numcol].dropna()
    if len(recent)>0:
        world_gov.loc[i, "CO2"]=dx[numcol].dropna().to_list()[-1]
slumwb=pd.read_csv(slumwb,skiprows=3)
numcol=slumwb.select_dtypes(include=numerics).columns
for i, dx in slumwb.set_index("Country Code").iterrows():
    
    recent=dx[numcol].dropna()
    if len(recent)>0:
        world_gov.loc[i, "slum_pop"]=dx[numcol].dropna().to_list()[-1]


world_gov["gdp/pop"]=world_gov["GDP_MD_EST"]/world_gov["POP_EST"]

###city
path="./data/avg3m_13_07_2023.json"
path="./data/uc_avg3m_14_07_2023.json"
city_res=gpd.read_file(f"{path}")
city_res["Cntry_ISO"]=city_res["CTR_MN_ISO"]
city_res["FUA_p_2015"]=city_res["P15"]
categories=["ECONOMY", "INCOME_GRP", 
            "CONTINENT", "REGION_UN", 
            "SUBREGION", "REGION_WB", 
            "developing", "developed", 
            "high_income","Region Name",
            "GDP_MD_EST", "slum_pop",
            
            ]
for i, dx in world_gov[categories].iterrows():
    ci=city_res[city_res["Cntry_ISO"]==i].index
    for c in ci:
        dx
        for di, d in zip(dx.index,dx):
            city_res.loc[c, di.strip()]=d
    

#shape of cities

"""
city_res["squarec"]=momepy.SquareCompactness(city_res).series
city_res["rectangu"]=momepy.Rectangularity(city_res).series
city_res["squarec"]=momepy.SquareCompactness(city_res).series
city_res["convex"]=momepy.Convexity(city_res).series
city_res["circ"]=momepy.CircularCompactness(city_res).series
city_res["elongat"]=momepy.Elongation(city_res).series
city_res["frac_ar"]=momepy.FractalDimension(city_res).series
"""
def rsquare(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()
    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    rsq = ssreg / sstot
    return rsq

def sami(y,x,m,b):
    return y -  (x*m + b)

def fitfunc(x,y):
    m ,b = np.polyfit(x,y,1)
    r2 = rsquare(x,y,1)
    return m,b,r2

def func(m,b, newX):
    
    return newX*m + b


def scale(ser):
    ser -= ser.min()
    ser /= ser.max()
    return ser.values 
def list_files(dir,ex="csv"):
   import os
   r=[]
   for root, dirs, files in os.walk(dir):
        for name in files:
            if name.split(".")[-1]==ex:
                r.append(os.path.join(root, name))
   print(f"generating data from {len(r)} files")
   return r


    
def _ccfplot(x):
    x=x.to_numpy()
    weights = np.ones_like(x)
    
    sorter = x.argsort()
    x = x[sorter]
    weights = weights[sorter]
    y = weights.cumsum()
    x = np.r_[-np.inf, x]
    y = np.r_[0, y]
    y = y.max() - y
    return (x, y)
    #plt.plot(x, y,ls, markersize=3, label=label)
   # plt.xscale('log')
    #plt.yscale('log')

city_res
city_res=city_res[city_res["avgh"]>0]
regions=city_res.groupby("Cntry_ISO")
negaras=pd.DataFrame()
for reg, _res in regions:
    if len(_res)>10:
        x, y=np.log10(_res["FUA_p_2015"]), np.log10( _res["avgh"])
        m, b, r2 = fitfunc(x,y)
        negaras.loc[reg, "m"]=m
        negaras.loc[reg, "r2"]=r2
        negaras.loc[reg, "b"]=b
        negaras.loc[reg, "avg_avgh"]=_res["avgh"].mean()
        negaras.loc[reg, "avg_avgpop3d"]=_res["avgpop3d"].mean()
        negaras.loc[reg, "avg_infpercap"]=_res["infpercap"].mean()

negaras=negaras.join(world_gov)
#negaras["slum_pop"]=negaras["slum_pop"].fillna(0)
hue_order=world_gov["developing"].dropna().unique()
#hue_order=world_gov["developed"].dropna().unique()
#hue_order=world_gov["high_income"].dropna().unique()

hue="developing"
fig, ax=plt.subplots(2, 2, sharey=True, figsize=(9, 5), )

#box plot of scaling per country grouped by region and development
ax=ax.flatten()
sns.boxplot(data=negaras, ax=ax[0],hue_order=hue_order,
                y="Region Name", x="m", 
                hue=hue,  palette=["w", "k"],
                medianprops=dict(color="red", alpha=0.7))
ax[0].axvline(negaras["m"].mean(), color="b", linestyle="--", zorder=0, label="global mean")
ax[0].set_xlabel('\u03B2')

sns.boxplot(y="Region Name", x="r2",
            hue=hue, palette=["w", "k"],
            hue_order=hue_order,
            medianprops=dict(color="red", alpha=0.7),
            data=negaras, ax=ax[1])
ax[1].axvline(negaras["r2"].mean(), color="b", linestyle="--", zorder=0, label="global mean")
ax[1].set_xlabel("$R^2$")

###box plot of h and 3dpop valeu per city group by region and development
_res=city_res.copy()
_res=_res[_res["Cntry_ISO"].isin(negaras.index)] # biar negara yg disample sama kyk kasus scaling per negara (min>10 kota dalam negara)

_res["avgpop3d"]=np.log10(_res["avgpop3d"])
sns.boxplot(y="Region Name", x="avgh",
            hue=hue, palette=["w", "k"],
            hue_order=hue_order,
            medianprops=dict(color="red", alpha=0.7),
            data=_res, ax=ax[2])
ax[2].axvline(_res["avgh"].mean(), color="b", linestyle="--", zorder=0, label="global mean")

sns.boxplot(y="Region Name", x="avgpop3d",
            
            hue=hue, palette=["w", "k"],
            hue_order=hue_order,
            medianprops=dict(color="red", alpha=0.7),
            data=_res, ax=ax[3])
ax[3].axvline(_res["avgpop3d"].mean(), color="b", linestyle="--", zorder=0, label="global mean")


for a in ax[:-1]:
    a.legend([],[], frameon=False)
for a in ax:
    a.set_ylabel("")
plt.legend()
plt.tight_layout()
plt.savefig("test1.png", dpi=200)
plt.show()


mark=matplotlib.lines.Line2D.markers
mark=pd.Series(mark)
mark=mark.sample(frac=1)
mark=list(mark[mark!="nothing"].index)

####inset pnas (avg=avg of avgh)
#possibly u-shape relation (gdp or co2 negara,  beta)
fig, ax=plt.subplots(1, 3, figsize=(10, 4), sharey=True)
for d in negaras.groupby("developing"):
    ax[0].scatter(d[-1]["gdp/pop"], d[-1]["avg_avgh"], label=d[0])
    
    ax[1].scatter( d[-1]["CO2"],d[-1]["avg_avgh"], label=d[0])
    ax[2].scatter(d[-1]["slum_pop"],d[-1]["avg_avgh"], #marker=mark[i],
                 
                  label=d[0])
    ax[0].set_xlabel("gdp/pop")
    ax[1].set_xlabel("CO2")
    ax[2].set_xlabel("Slum pop")
plt.legend()
ax[0].set_ylabel("avg h")
plt.savefig("test2.png", dpi=200)
plt.show()

fig, ax=plt.subplots(1, 3, figsize=(10, 4), sharey=True)
for d in negaras.groupby("developing"):
    ax[0].scatter(d[-1]["gdp/pop"], d[-1]["avg_avgpop3d"], label=d[0])
    
    ax[1].scatter( d[-1]["CO2"],d[-1]["avg_avgpop3d"], label=d[0])
    ax[2].scatter(d[-1]["slum_pop"],d[-1]["avg_avgpop3d"], #marker=mark[i],
                 
                  label=d[0])
    ax[0].set_xlabel("gdp/pop")
    ax[1].set_xlabel("CO2")
    ax[2].set_xlabel("Slum pop")
plt.legend()
ax[0].set_ylabel("avg 3dpop")
plt.savefig("test2_1.png", dpi=200)
plt.show()

fig, ax=plt.subplots(1, 3, figsize=(10, 4), sharey=True)
for d in negaras.groupby("developing"):
    ax[0].scatter(d[-1]["gdp/pop"], d[-1]["avg_infpercap"], label=d[0])
    
    ax[1].scatter( d[-1]["CO2"],d[-1]["avg_infpercap"], label=d[0])
    ax[2].scatter(d[-1]["slum_pop"],d[-1]["avg_infpercap"], #marker=mark[i],
                 
                  label=d[0])
    ax[0].set_xlabel("gdp/pop")
    ax[1].set_xlabel("CO2")
    ax[2].set_xlabel("Slum pop")
plt.legend()
ax[0].set_ylabel("avg_infpercap")
plt.savefig("test2_1.png", dpi=200)
plt.show()



fig, ax=plt.subplots(1, 3, figsize=(10, 4), sharey=True)
for d in negaras.groupby("developing"):
    ax[0].scatter(d[-1]["gdp/pop"], d[-1]["m"], 
                  
                  label=d[0])
    ax[1].scatter(d[-1]["CO2"],d[-1]["m"], #marker=mark[i],
                 
                  label=d[0])
    ax[2].scatter(d[-1]["slum_pop"],d[-1]["m"], #marker=mark[i],
                 
                  label=d[0])
    ax[0].set_xlabel("gdp/pop")
    ax[1].set_xlabel("CO2")
    ax[2].set_xlabel("Slum pop")
plt.legend()
ax[0].set_ylabel("scaling")
plt.savefig("test3.png", dpi=200)
plt.show()


##### peta dunia 

def petadunia(negaradf, col, legend=True,
              legend_kwds={
       "orientation":"horizontal",
       "pad":0.02,
       "shrink":.5315},
        missing_kwds={'color': 'lightgrey',
                      "hatch":'\\\\\\\\'},
        linewidth=.8,
        cmap="hot"
       ):
    #world_gov as fix geoms layer paling bawah
    plotdf=world_gov.join(negaradf[col])
    #ax=world_gov.plot(facecolor="None",  linewidth=.5)
    
    ax=plotdf.plot(col, legend=legend, linewidth=linewidth,
                legend_kwds=legend_kwds,missing_kwds=missing_kwds, cmap=cmap)
    ax.axis("off")
    return ax


col="m"
petadunia(negaras, col, legend=True, cmap="magma")









negaras.columns
regions=city_res.groupby("Region Name")

fig, axs=plt.subplots(2,2, figsize=(10, 10))
axs=axs.flatten()
limitvar="infpercap"
limits=city_res[limitvar]
color=city_res["slum_pop"].fillna(0)
#norm = matplotlib.colors.LogNorm(vmin=color.min(), vmax=color.max())
norm = matplotlib.colors.Normalize(vmin=color.min(), vmax=color.max())
cm=vars(matplotlib.cm)["magma"]
for i, (reg, res) in enumerate(regions):
    ax=axs[i]
    #_res=_res.query(f"{limitvar}>{limit}")
    for ctr, _res in  res.groupby("Cntry_ISO"):
        if len (_res)>10:
            cval=color.loc[_res.index].unique().max()
            colorhex=matplotlib.colors.rgb2hex(cm(norm(cval), bytes=False))
            x, y=np.log10(_res["FUA_p_2015"]), np.log10( _res["avgh"])
            m, b, r2 = fitfunc(x,y)
            newX = np.linspace(np.min(x),np.max(x))
            labs= r"$\beta=${0:9.3f}, ".format(m) + r" $R^2=${0:9.3f}".format(r2)
            ax.plot(newX, func(m,b, newX),"-" ,
                   label =reg+":"+labs,
                   color= colorhex
                   #color="#1e9b8a"
                   )
            
            ax.scatter(np.log10(_res["FUA_p_2015"]), np.log10(_res["avgh"]), 
                       alpha=0.35, 
                       color= colorhex,
                       s=8)
            
    ax.set_title(reg)
    #ax.legend(loc="lower right")
#plt.legend()
plt.tight_layout()
plt.savefig("limitbypop_developing.pdf")



"""

figall, axall=plt.subplots(1)
fig, ax=plt.subplots(1)
cm = plt.get_cmap('Accent')
color=[cm(1.*i/len(regions)) for i in range(len(regions))]

for (g, plotdata), c in zip(regions, color):
    print(g)
  
    
    files=list_files(dir="cell_files", ex="gz")
    files=pd.Series(files)
    isin=files.apply(lambda x: float(x.split("cell_")[-1].split(".gz")[0])).isin(plotdata.eFUA_ID)

    P=[]
    H=[]
    D=[]
    for i, path in enumerate(files[isin]):

        #cell=pd.read_csv(f, index_col=0)
        with open(path, "rb") as f:
            compressed_data = f.read()
            decompressed_data = gzip.decompress(compressed_data)
        
        data = pickle.loads(decompressed_data)
        #print(data)
        cell = gpd.GeoDataFrame.from_dict(data)
        P.append(cell["Pval"])
        H.append(cell["Hval"])
        D.append(cell["3d_dens"])
        

        
        x, y=_ccfplot(cell["Hval"])
        ax.plot(x, y, "o-", markersize=1.3, color=c, label=g if i == 0 else "", alpha=0.6,)
        #sns.kdeplot(cell["Hval"], ax=ax, color=c, label=g if i == 0 else "", alpha=0.5, )
    

    
    x, y=_ccfplot(pd.concat(H).reset_index(drop=True))
    axall.plot(x, y, "o-", markersize=1.3, color=c, label=g, alpha=0.6,)
    #pd.concat(P).plot.kde(ax=axall, label=g)
axall.set_xscale('log')
axall.set_yscale('log')   
axall.set_xlabel("h val")
axall.set_ylabel("freq")  
figall.legend()
ax.set_xscale('log')
ax.set_yscale('log')    
ax.set_xlabel("h val")
ax.set_ylabel("freq")  
fig.legend()
fig.savefig("cellstats1.jpg", dpi=200, bbox_inches="tight")
figall.savefig("cellstats1.jpg", dpi=200, bbox_inches="tight")
    
"""









