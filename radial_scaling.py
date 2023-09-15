#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 20:42:02 2023

@author: genta
"""
import itertools
import geopandas as gpd
import numpy as np
from scipy.spatial import distance
import osmnx
import pandas as pd
import matplotlib.pyplot as plt

def radial_center_point (start, num, data):
    target=[(x, y) for x, y in zip(data.x, data.y)]
    center=data.dissolve().centroid
    dismat=distance.cdist([(center.x[0], center.y[0])], target).T
    first=start
    for dist in np.arange(start, dismat.max(), num):
        #print(dist)
        where=np.where((dismat>first) & (dismat<dist))
        local=dismat[where]
        data.loc[where[0], "ovr"]=str(first)+"-"+str(dist)
        first=dist
    where=np.where((dismat>first))
    data.loc[where[0], "ovr"]=str(first)+"-"+str(first+num)
    return data
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

def fitfunc(x,y):
    m,b = np.polyfit(x,y,1)
    r2 = rsquare(x,y,1)
    return m,b,r2

def list_files(dir,ex="csv"):
   import os
   r=[]
   for root, dirs, files in os.walk(dir):
        for name in files:
            if name.split(".")[-1]==ex:
                r.append(os.path.join(root, name))
   print(f"generating data from {len(r)} files")
   return r

def radial_scaling_all(datas, y_col, start, num):
    glob={}
    betas={}
    for data in datas:
        print(data)
        data=gpd.read_file(data)
        data=radial_center_point (start, num, data)
        grp=data.groupby("ovr")
        local=grp[[y_col, "Pval"]].sum()
        for i in local.index:
            if i in glob.keys():
                glob[i]=glob[i]+[local.loc[i]]
            else:
                glob[i]=[local.loc[i]]
                
    for k, v in glob.items():
        if len(v)>100:
            plotd=pd.concat([  pd.DataFrame(d).T for d in v]).reset_index()
            plotd=plotd.dropna()
            plotd=plotd.query(f"{y_col} > 0 and Pval>0")
            x,y=np.log10(plotd[y_col]), np.log10(plotd["Pval"])
            betas[k]=fitfunc(x,y)
    betas=pd.DataFrame(betas).T
    betas.columns =["m", "b", "r2"]    
    return betas 
if __name__ == '__main__':

    start, num=0, 1000
    datas=list_files("/Volumes/HDPH-UT/K-Jkt copy/cbd-slum/data/cell_files/", "json")
    y_col="Hval"



    betas=radial_scaling(datas, y_col, start, num)
    betas["m"].plot()
    plt.show()
