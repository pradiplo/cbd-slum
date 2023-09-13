import os
import matplotlib.pyplot as plt
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import matplotlib as mpl
import pandas as pd
import numpy as np
import pickle
import gzip
#mpl.style.use('classic')
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
project='EPSG:4326'

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

def read_compressed(path): #buat buka file .gz di atas wk
 
    with open(path, 'rb') as f:
        compressed_data = f.read()
        
    
    # Decompress data using gzip and pickle
        decompressed_data = gzip.decompress(compressed_data)
    
    
   # import pygeos as sp
    
    data = pickle.loads(decompressed_data)
        
   
        #print(data)
    gdf = gpd.GeoDataFrame.from_dict(data)

    return gdf


df_small_path = "/Users/tagawayo-lab-pc43/workspace/urban/cbd-slum/data/avg3m_small_24_07_2023.json"
df_large_path =  "/Users/tagawayo-lab-pc43/workspace/urban/cbd-slum/data/avg3m_large_24_07_2023.json"
volume_path = "/Users/tagawayo-lab-pc43/workspace/urban/cbd-slum/data/builtvolume_31_07_2023.json"
city_s=gpd.read_file(df_small_path)
city_l=gpd.read_file(df_large_path)
city = gpd.GeoDataFrame( pd.concat([city_l,city_s])).set_index("ID_HDC_G0",drop=False)
city_v = gpd.read_file(volume_path).set_index("ID_HDC_G0")

city_v = city_v[city_v["totvol"] > 0  ]
#city_v = city_v[city_v["P15"] > 1e5  ]
#print(len(city_l))
#print(len(city_s))
#print(len(city))
#city = city[city["P15"] > 1e5]
#print(city.columns)
#print(city["infpercap"].max())
#print(city["avgh"].mean())
#jkt = city[city["UC_NM_MN"] == "Jakarta"]
#print(jkt["ID_HDC_G0"])
#cellHP = read_compressed("./data/cell_files_uc/cell_"  + str(jkt.ID_HDC_G0.values[0]) + ".gz")
#norm=matplotlib.colors.LogNorm(vmin=0.001,vmax=100)
#print(cellHP["Pval"].sum())
#print(jkt["P15"])
#city["sprdif"] = city["spr_pop"] - city["spr_h"]
#city["sprdif_abs"] = np.abs(city["spr_pop"] - city["spr_h"])

avg_spr_p = city["spr_pop"].mean()
avg_spr_h = city["spr_h"].mean()

city["h_poly"] = city["spr_h"] > avg_spr_h
city["p_poly"] = city["spr_pop"] > avg_spr_p

truedic = {True: "Poly", False: "Mono"}

city["h_poly"] = city["h_poly"].map(truedic)
city["p_poly"] = city["p_poly"].map(truedic)

#print(city["h_poly"])

#city.plot.hist(column="infpercap",by="p_poly", figsize=(10, 8))
#plt.show()


polys = city[city["p_poly"] == "Poly" ]
monos = city[city["p_poly"] == "Mono"]


mm, bm, r2m = fitfunc(np.log10(city_v["P15"]),np.log10(city_v["totvol"]))
#mp, bp, r2p = fitfunc(np.log10(polys["P15"]),np.log10(polys["avgh"]))

plt.scatter(np.log10(city_v["P15"]), np.log10(city_v["totvol"]),s=1)
plt.show()

print(mm)
print(bm)

#print(mp)
#print(bp)

plt.figure()
plt.scatter(city["P15"], city["spr_h"],s=5)
plt.xscale("log")
#plt.yscale("log")
#plt.ylim(-0.1,1)
plt.xlabel("Pop")
plt.ylabel("spr h")
#plt.legend()
plt.savefig("spr_h.png")

plt.figure()
plt.scatter(city["P15"], city["spr_pop"],s=5)
plt.xscale("log")
#plt.yscale("log")
#plt.ylim(-0.1,1)
plt.xlabel("Pop")
plt.ylabel("spr p")
#plt.legend()
plt.savefig("spr_p.png")
"""
plt.figure()
sc = plt.scatter(city["spr_pop"],city["spr_h"], s=5,c=city["P15"], norm=mpl.colors.LogNorm(vmin=1e5,vmax=1e8))
plt.colorbar(sc)
#plt.xscale("log")
#plt.yscale("log")
#plt.ylim(-0.1,1)
plt.xlabel("Spr pop")
plt.ylabel("Spr H")
#plt.legend()
plt.savefig("sprh-vs-sprpop.png")

plt.figure()
plt.scatter(city["P15"], city["sprdif"],s=5)
plt.xscale("log")
#plt.yscale("log")
#plt.ylim(-0.1,1)
plt.xlabel("Pop")
plt.ylabel("Spr pop - spr h")
#plt.legend()
plt.savefig("sprdifvspop.png")

plt.figure()
plt.scatter(city["P15"], city["sprdif_abs"],s=5)
plt.xscale("log")
#plt.yscale("log")
#plt.ylim(-0.1,1)
plt.xlabel("Pop")
plt.ylabel("Spr pop - spr h")
#plt.legend()
plt.savefig("sprdifvspop_abs.png")
"""
#tolong dibikin lebih fancy wk