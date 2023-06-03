from osgeo import gdal
from osgeo import ogr
import rasterio
import rasterio.mask as RM
import matplotlib.pyplot as plt
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import osmnx 
import shutil
import pandas as pd
import numpy as np
from pysal.explore import inequality
from joblib import Parallel, delayed
from tqdm import tqdm

pd.options.mode.chained_assignment = None 

gdal.UseExceptions()

project='EPSG:4326'

def shaderxy_plot(df, x, y, z, cmap, how="log"):
    import datashader as ds, pandas as pd, colorcet
    from matplotlib import cm
    #df  = pd.read_csv('census.csv')
    cvs = ds.Canvas(plot_width=850, plot_height=500)
    agg = cvs.points(df, x, y, agg=ds.mean(z))
    img = ds.tf.shade(agg, cmap=vars(cm)[cmap], how=how)
    return img

def get_raster_crs(file):
    with rasterio.open(file) as src:
        return src.crs

def mask_raster(rasfile, shape_geoser,i=""):
    with rasterio.open(rasfile) as src:
        crs=src.crs
        shape_geoser=shape_geoser.to_crs(crs)
        shape=shape_geoser.values
        out_image, out_transform = RM.mask(src, shape, crop=True, nodata=np.nan)
        out_meta = src.meta
        out_meta.update({"driver":"GTiff",
                         "height":out_image.shape[1],
                         "width":out_image.shape[2],
                         "transform": out_transform})
    with rasterio.open("masked_"+str(i)+".tif", "w+", **out_meta) as dest:
        dest.write(out_image)
    return "masked_"+str(i)+".tif"

def pixelOffset2coord(raster, xOffset,yOffset):
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    coordX = originX+pixelWidth*xOffset
    coordY = originY+pixelHeight*yOffset
    return coordX, coordY

def raster2array(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    array = band.ReadAsArray()
    return array

def check_pixelANDcell(rasterfn, outSHPfn):
    d=len(gpd.read_file(outSHPfn))
    raster = raster2array(rasterfn)
    px=raster.shape[1]*raster.shape[0]
    print(d, px)
    print("pixel pont and cell geometry have same len?->", (d-px)==0)

def array2shp(array,outSHPfn,rasterfn):
    # max distance between points
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    pixelWidth = geotransform[1]
    
    # wkbPoint
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outSHPfn):
        shpDriver.DeleteDataSource(outSHPfn)
    outDataSource = shpDriver.CreateDataSource(outSHPfn)
    outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint )
    featureDefn = outLayer.GetLayerDefn()
    outLayer.CreateField(ogr.FieldDefn("VALUE", ogr.OFTInteger))

    # array2dict
    point = ogr.Geometry(ogr.wkbPoint)
    row_count = array.shape[0]
    for ridx, row in enumerate(array):
        #if ridx % 100 == 0:
            #print ("{0} of {1} rows processed".format(ridx, row_count))
        for cidx, value in enumerate(row):
            Xcoord, Ycoord = pixelOffset2coord(raster,cidx,ridx)
            point.AddPoint(Xcoord, Ycoord)
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(point)
            outFeature.SetField("VALUE", float(value))
            outLayer.CreateFeature(outFeature)
            outFeature.Destroy()
  #  outDS.Destroy()

def main_raster2polypoint(rasterfn,outSHPfn, bufferm=50):
    array = raster2array(rasterfn)
    array2shp(array,outSHPfn,rasterfn)
    d=gpd.read_file(outSHPfn)
    d=d.set_crs( get_raster_crs(rasterfn))
    if d.crs.is_projected == False:
        d=osmnx.project_gdf(d)
    d["x"]=d.geometry.x
    d["y"]=d.geometry.y
    d['geometry']=d.buffer(bufferm, cap_style = 3)
    return d

def citywise_job(rasfile, geoser, cityID="", removefile=True, check_pixel=True):
     rasterfn=mask_raster(rasfile, geoser, i=cityID)
     outSHPfn="polygonized_"+str(cityID)
     cell=main_raster2polypoint(rasterfn, outSHPfn)
     #cell=cell[cell.VALUE!=-1]
     
     if check_pixel:
         check_pixelANDcell(rasterfn, outSHPfn)
     
     if removefile:
         shutil.rmtree(outSHPfn)
         os.remove(rasterfn)
    
     return cell

def get_gini(gdf,col_name):
    g15 = inequality.gini.Gini(gdf[col_name].values)
    gmax = (len(gdf) - 1) / len(gdf)
    gini = g15.g / gmax
    return gini

def check_cell_diff(cell1,cell2):
    xdiff=((cell1.x-cell2.x)==0).all()
    ydiff=((cell1.y-cell2.y)==0).all()
    xymatch=pd.Series([xdiff, ydiff]).all()
    lenmatch= len(cell1)==len(cell2)
    equals = xymatch and lenmatch
    return equals

def mse(arr1, arr2):
   diff = arr1 - arr2
   err = np.sum(diff**2)
   mse = err/(float(len(arr1)))
   return mse

def scale(ser):
    ser -= ser.min()
    ser /= ser.max()
    return ser.values 

def get_difference(gdf,col1,col2):
    arr_1 = scale(gdf[col1])
    arr_2 = scale(gdf[col2])
    return mse(arr_1,arr_2)

def mean_center(points, w=None):
    #points==> array([-000x, 000y])
    points=np.asarray(points)
    if w is not None:
        w=np.asarray(w)
        w = w * 1.0 / w.sum()
        w.shape = (1, len(points))
        return np.dot(w, points)[0]
    else:
        return points.mean(axis=0)

def get_cell(cityID):
    name=city[city.eFUA_ID==cityID].eFUA_name
    #print("city id", (cityID,name.values[0]))
    geoser=city[city.eFUA_ID==cityID].geometry
    ####### intinya!!
    cellH=citywise_job(h, geoser, cityID, removefile=True, check_pixel=False)
    cellP=citywise_job(p, geoser, cityID, removefile=True, check_pixel=False)
    ######
    eqs = check_cell_diff(cellH,cellP)
    if eqs:
        #merging cellH and CellP = cellHP 
        cellHP=cellH.join(cellP.VALUE.rename("P"))
        cellHP.columns=["Hval", "geometry", "x", "y","Pval"]
        cellHP=cellHP[cellHP.Hval>=0] #nan value entah knp jadi -2147483648 use H sebagai destinasi
        cellHP.loc[cellHP["Pval"] <0, "Pval"] = 0
        cellHP["3d_dens"] = cellHP["Pval"] / cellHP["Hval"]  
        return cellHP
    else:
        print(cityID, "has different h and p cropped-raster data")
        return 0

def calc_aggregate(cityID):
    cellHP = get_cell(cityID)
    avgh = cellHP["Hval"].mean()
    h_thrs = [25,35,45,55,65]
    cbds = [cellHP[cellHP["Hval"] >= h_thr] for h_thr in h_thrs ]
    cbd_areas = [len(cbd) * 1e-2 for cbd in cbds] #in km2
    mse_pop_h = get_difference(cellHP,"Pval","Hval")
    mse_2d_3d = get_difference(cellHP,"Pval","3d_dens")
    gini_pop = get_gini(cellHP,"Pval")
    gini_h = get_gini(cellHP,"Hval")
    gini_3dpop = get_gini(cellHP, "3d_dens")
    avgpop3d = cellHP["3d_dens"].mean()
    maxpop3d = cellHP["3d_dens"].max()
    minpop3d = cellHP["3d_dens"].min()# meangless make it non-zore min??
    return (cityID, cbd_areas, gini_pop, gini_h, gini_3dpop, avgpop3d, maxpop3d, minpop3d, avgh, mse_pop_h, mse_2d_3d)

def print_file(cityID):
    cellHP = get_cell(cityID)
    cellHP.to_file("./data/cell_files/cell_"  + str(cityID) + ".json",driver="GeoJSON")
    return 0

#ok
ghsfua="./data/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
h="./data/GHS_BUILT_H_ANBH_E2018_GLOBE_R2022A_54009_100_V1_0/GHS_BUILT_H_ANBH_E2018_GLOBE_R2022A_54009_100_V1_0.tif"
p="./data/GHS_POP_E2020_GLOBE_R2022A_54009_100_V1_0/GHS_POP_E2020_GLOBE_R2022A_54009_100_V1_0.tif"

city=gpd.read_file(ghsfua)
city = city.sort_values("FUA_p_2015").tail(200)
city = city.set_index("eFUA_ID",drop=False)
city = city.sample(n=60)

#for running in cluster with SLURM

num_cores= int(os.environ['SLURM_CPUS_PER_TASK'])

print("Working with " +str(num_cores) + " cores for " + str(len(city)) + " cities")

results = Parallel(n_jobs=num_cores, verbose=1)(delayed(print_file)(idx) for (idx) in tqdm(city.eFUA_ID))

"""
results = Parallel(n_jobs=num_cores, verbose=1)(delayed(calc_aggregate)(idx) for (idx) in tqdm(city.eFUA_ID))
city_res = city.join(pd.DataFrame(results, columns=["index","cbd_areas","gini_pop","gini_h","gini_3dpop","avgpop3d","maxpop3d","minpop3d","avgh","jcrd_pop_h","jcrd_2d_3d"]).set_index("index"))
city_res = city_res.drop("eFUA_ID",axis=1)
city_res[["cbd_a_25","cbd_a_35","cbd_a_45","cbd_a_55","cbd_a_65"]] = pd.DataFrame(city_res.cbd_areas.to_list(), index=city_res.index)
city_res = city_res.drop("cbd_areas",axis=1)

print(city_res)
city_res.to_file("./data/hthr25-65.json",driver="GeoJSON")
"""

