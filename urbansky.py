#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:54:47 2023

@author: kunogenta
"""
import matplotlib.pyplot as plt
from datetime import date
import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
import seaborn as sns
#botanical_continents=gpd.read_file("data/level1.geojson")#https://github.com/tdwg/wgsrpd/blob/master/geojson/level1.geojson
world_gov=pd.read_csv("data/UNSD — Methodology.csv",error_bad_lines=False, sep=";")
#https://unstats.un.org/unsd/methodology/m49/overview/
world_gov=world_gov.set_index("ISO-alpha3 Code")
world_gov=world_gov[["Sub-region Name", "Region Name", 
           "Least Developed Countries (LDC)",                                  
           "Land Locked Developing Countries (LLDC)",                           
           "Small Island Developing States (SIDS)"]
          ]
today = date.today().strftime("%d_%m_%Y")
city_res=gpd.read_file(f"./data/hthr_{today}.json")



for i, dx in world_gov.iterrows():
    ci=city_res[city_res["Cntry_ISO"]==i].index
    for c in ci:
        for di, d in zip(dx.index,dx):
            city_res.loc[c, di.strip()]=d
    


#shape of cities
city_res["rectangu"]=momepy.Rectangularity(city_res).series
city_res["squarec"]=momepy.SquareCompactness(city_res).series
city_res["convex"]=momepy.Convexity(city_res).series
city_res["circ"]=momepy.CircularCompactness(city_res).series
city_res["elongat"]=momepy.Elongation(city_res).series
city_res["frac_ar"]=momepy.FractalDimension(city_res).series


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



regions=city_res.groupby("Region Name")

plt.scatter( city_res["circ"], city_res["spr_h"])
plt.xlabel("circular compactness")
plt.ylabel("spr_h")
plt.savefig("test.jpg", dpi=200)
plt.show()
plt.scatter( city_res["circ"], city_res["gini_h"])
plt.xlabel("circular compactness")
plt.ylabel("gini_h")
plt.show()


plt.scatter(city_res["FUA_p_2015"], city_res["avgh"])
plt.xscale("log")
plt.yscale("log")
plt.show()

fig, ax=plt.subplots(1)
cm = plt.get_cmap('Accent')
color=[cm(1.*i/len(regions)) for i in range(len(regions))]

for (g, plotdata), c in zip(regions, color):
    print(g)
  
    
    files=list_files(dir="data/cell_files")
    files=pd.Series(files)
    isin=files.apply(lambda x: float(x.split("cell_")[-1].split(".csv")[0])).isin(plotdata.eFUA_ID)
    P=[]
    H=[]
    D=[]
    for i, f in enumerate(files[isin]):
        cell=pd.read_csv(f, index_col=0)
        P.append(cell["Pval"])
        H.append(cell["Hval"])
        D.append(cell["3d_dens"])
        x, y=_ccfplot(cell["Hval"])
        ax.plot(x, y, "o-", markersize=0.3, color=c, label=g if i == 0 else "", alpha=0.6,)
        #sns.kdeplot(cell["Hval"], ax=ax, color=c, label=g if i == 0 else "", alpha=0.5, )
plt.xscale('log')
plt.yscale('log')    
plt.xlabel("h val")
plt.ylabel("freq")  
leg=plt.legend()
plt.savefig("cellstats1.jpg", dpi=200, bbox_inches="tight")
    










