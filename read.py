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
from pysal.lib import cg
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import date
today = date.today().strftime("%d_%m_%Y")
pd.options.mode.chained_assignment = None 

gdal.UseExceptions()

project='EPSG:4326'

def shaderxy_plot(df, x, y, z, cmap, how="log"):
    import datashader as ds
    from matplotlib import cm
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

def getXY(pt):
    return (pt.x, pt.y)

def get_eta(gdf,col_name):
    gdf=gdf.copy()
    gdf=gdf.reset_index(drop=True)
    thres =  gdf[col_name].mean() #or loubar, tp mean lbh oke kayanya
    centroidseries = gdf["geometry"].centroid
    x, y = [list(t) for t in zip(*map(getXY,centroidseries))]
    matpos = np.array([x,y]).transpose()
    d = cg.distance_matrix(matpos)
    eucl = np.maximum(d, d.transpose())
    gdf_dict = gdf[col_name].to_dict()
    s = [(k, gdf_dict[k]) for k in sorted(gdf_dict, key=gdf_dict.get)]
    keys = []
    vals = []
    for k,v in s:
        keys.append(k)
        vals.append(v)
    vals = np.array(vals)
    keys = np.array(keys)
    loubar_keys = keys[vals>=thres]
    dist_mat =  eucl[keys.reshape(-1,1), keys]
    dist_corr = dist_mat[dist_mat>0]
    loubar_dist_mat = eucl[loubar_keys.reshape(-1,1), loubar_keys]
    loubar_dist_corr = loubar_dist_mat[loubar_dist_mat>0]
    eta = loubar_dist_corr.mean()/dist_corr.mean()
    return eta

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
    
def calc_loubar_threshold(gdf,raster_val):
        lourank = int(len(gdf)*(1 - gdf[raster_val].mean()/max(gdf[raster_val])))
        gdf_rank = gdf.sort_values(by=[raster_val],ascending=True).reset_index(drop=True)
        return gdf_rank.loc[lourank][raster_val]

def get_jcr_hot(gdf, col1, col2):
    from sklearn.metrics import jaccard_score as js
    gdf=gdf.astype({col1:float, col2:float})
    ar1=(gdf[col1]>= gdf[col1].mean()).replace({True:1, False:0})
    ar2=(gdf[col2]>= gdf[col2].mean()).replace({True:1, False:0})
    return js(ar1.to_numpy(), ar2.to_numpy())


def get_cell(cityID):
    #name=city[city.eFUA_ID==cityID].eFUA_name
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
        cellHP["3d_dens"] = cellHP["Pval"] / cellHP["Hval"]  #can be 0-div probelm,yields inf val ? == make gini and mse nan? hrs dihandle
        return cellHP
    else:
        print(cityID, "has different h and p cropped-raster data")
        return 0

def calc_aggregate(cityID,   h_thrs = [15,25,35,45,55,65]):
    vars2exclude=["vars2exclude", "cellHP", "h_thrs","cbds" ]
    cellHP = get_cell(cityID)
    
    
    cbds = [cellHP[cellHP["Hval"] >= h_thr] for h_thr in h_thrs ]
    cbd_areas = [len(cbd) * 1e-2 for cbd in cbds] #in km2
    #get diff 

    mse_pop_h = get_difference(cellHP,"Pval","Hval")
    mse_2d_3d = get_difference(cellHP[cellHP["3d_dens"].notna()],"Pval","3d_dens")

    jcr_pop_h  =  get_jcr_hot(cellHP, "Pval", "Hval")
    jcr_2d_3d =  get_jcr_hot(cellHP[cellHP["3d_dens"].notna()], "Pval", "3d_dens")

    gini_pop = get_gini(cellHP,"Pval")
    gini_h = get_gini(cellHP,"Hval")
    gini_3dpop = get_gini(cellHP[cellHP["3d_dens"].notna()], "3d_dens")

    spr_pop = get_eta(cellHP,"Pval")
    spr_h = get_eta(cellHP,"Hval")
    spr_3dpop = get_eta(cellHP[cellHP["3d_dens"].notna()], "3d_dens") 

    avgpop3d,  avgh  = cellHP["3d_dens"].mean(), cellHP["Hval"].mean()
    maxh, maxpop3d    = cellHP["3d_dens"].max(), cellHP["Hval"].max()
    
    local_vars=locals() # a dict of local variable in this function 
                        #(at this line so far)
    rets= {k: v for k, v in local_vars.items() if k not in vars2exclude}
    #make return variable (kecuali yang string namenya ada di vars2exclude)
    #next time gausah make col name "string" yg  mendokusai 
    #col name in dataframe of city_res bakal just as defined in this def !!!
    return rets

def print_file(cityID):
    cellHP = get_cell(cityID)
    cellHP.to_file("./data/cell_files/cell_"  + str(cityID) + ".json",driver="GeoJSON")
    return 0

#ok
ghsfua="./data/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
h="./data/GHS_BUILT_H_ANBH_E2018_GLOBE_R2022A_54009_100_V1_0.tif"
p="./data/GHS_POP_E2015_GLOBE_R2022A_54009_100_V1_0.tif"



city = gpd.read_file(ghsfua)
city = city.sort_values("FUA_p_2015").sample(100)
city = city.set_index("eFUA_ID",drop=False)


#for running in cluster with SLURM
#num_cores= int(os.environ['SLURM_CPUS_PER_TASK'])

num_cores=-1
h_thrs=[15,25,35,45,55,65] #define h trsh first 
#(biar bisa pake komprehensi list dan gak nulis2 lagi pas nge-wide list of areas)
print("Working with " +str(num_cores) + " cores for " + str(len(city)) + " cities")

results = Parallel(n_jobs=num_cores, verbose=1)(delayed(calc_aggregate)(idx, h_thrs) for (idx) in tqdm(city.eFUA_ID))

city_res = city.join(pd.DataFrame(results).set_index("cityID"))
city_res = city_res.drop("eFUA_ID",axis=1)
city_res[["cbd_a_"+str(t) for t in h_thrs]] = pd.DataFrame(city_res.cbd_areas.to_list(), index=city_res.index)
city_res = city_res.drop("cbd_areas",axis=1)
city_res.to_file(f"./data/hthr_{today}.json",driver="GeoJSON")

