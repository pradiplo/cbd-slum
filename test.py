#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:16:24 2023

@author: kunogenta
"""

import geopandas as gpd
import esda
import momepy 
from pysal.lib import weights
import scipy
import pandas as pd
import matplotlib.pyplot as plt

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
def calc_loubar_threshold(gdf,raster_val):
    lourank = int(len(gdf)*(1 - gdf[raster_val].mean()/max(gdf[raster_val])))
    gdf_rank = gdf.sort_values(by=[raster_val],ascending=True).reset_index(drop=True)
    return gdf_rank.loc[lourank][raster_val]

def get_jcr_hot(gdf, col1, col2):
    from sklearn.metrics import jaccard_score as js
    gdf=gdf.astype({col1:float, col2:float})
    #gdf[col1]=gdf[col1].astype(float)
    #gdf[col2]=gdf[col2].astype(float)
    ar1=(gdf[col1]>= calc_loubar_threshold(gdf, col1)).replace({True:1, False:0})
    ar2=(gdf[col2]>= calc_loubar_threshold(gdf, col2)).replace({True:1, False:0})
    return js(ar1.to_numpy(), ar2.to_numpy())

save=[]
for f in list_files("/Volumes/HDPH-UT/K-Jkt copy/cell_files/", "json"):
    
    try:
        cellHP=gpd.read_file(f)
    except:
        cellHP=None
    
    if cellHP is not None:
        print(cellHP["3d_dens"].max())
        jcr_pop_h  =  get_jcr_hot(cellHP, "Pval", "Hval")
        jcr_2d_3d =  get_jcr_hot(cellHP[cellHP["3d_dens"].notna()], "Pval", "3d_dens")
        cor=scipy.stats.pearsonr(scale(cellHP["Pval"]), scale(cellHP["Hval"]))[0]
        save.append((cor, jcr_pop_h, jcr_2d_3d))
#pd.Series(w.neighbors).apply(lambda x: scipy.stats.iqr(data.loc[x]["Pval"]))
data=pd.DataFrame(save, columns=["c", "jph", "j23"])


plt.scatter(data["j23"],data["jph"])
plt.ylabel("j h dan p")
plt.xlabel("j 2d dan 3d")
