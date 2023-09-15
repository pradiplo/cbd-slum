from osgeo import gdal
from osgeo import ogr
import rasterio
import rasterio.mask as RM
#import matplotlib.pyplot as plt
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import osmnx 
import shutil
import pickle
import gzip
import pandas as pd
import numpy as np
from pysal.explore import inequality
#from pysal.lib import cg
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import date
today = date.today().strftime("%d_%m_%Y")
pd.options.mode.chained_assignment = None 
import warnings
import timeit
import tracemalloc
from itertools import combinations
import sklearn
import matplotlib.pyplot as plt
import math
from barneshut import calc_avg_distance
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
        out_image, out_transform = RM.mask(src, shape, crop=True, nodata=-1)
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
    #print(d, px)
    #print("pixel pont and cell geometry have same len?->", (d-px)==0)

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
    d=d.set_crs(get_raster_crs(rasterfn))
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

def calculate_distance_matrix(points):
    from scipy.spatial import distance
    num_points = points.shape[0]
    dist_matrix = np.zeros((num_points, num_points))
    #print(num_points)
    for i in range(num_points):
        for j in range(i+1, num_points):
            #distance = np.linalg.norm(points[i] - points[j])
            #print(points[i])
            distance_val = distance.euclidean(points[i], points[j])
            #print(distance_val)
            dist_matrix[i, j] = distance_val
            #dist_matrix[j, i] = distance_val

    return dist_matrix + dist_matrix.T

def calculate_avg_dist_loop(points):
    from scipy.spatial import distance
    combos=combinations(points,2)
    num_points = points.shape[0]
    pair_num= num_points*(num_points-1)/2
    distances=sum(distance.euclidean(p[0], p[-1]) for p in combos) #sum (bla3) lebih cepat drpd sum(list) dan for loop
    return distances/pair_num


def calculate_avg_dist_chunk(points,njobs):
    from sklearn.metrics import pairwise_distances_chunked #returning generator of arr (i, .......j) daripada pdist arr (I, J) 
    num_points = points.shape[0]                   # sum of sum of each chunks == sum of full box/arr -> /2 = non dup pair sum
    def reduce_func(D_chunk, start):
        sums = [sum(d) for d in D_chunk]
        return sums
    pair_num= num_points*(num_points-1)/2
    p=pairwise_distances_chunked(points, reduce_func=reduce_func, n_jobs=njobs)
    distances=sum(sum(a) for a in p)/2  # setengah isi kotak= sum o unique pair dist 

    return distances/pair_num
       
def get_eta(gdf,col_name):
    #gdf=gdf.copy()
    gdf=gdf.reset_index(drop=True) # keys assumes 0->x ordered index tp gdf took the effect of value filterings 
    thres = calc_loubar_threshold(gdf,col_name)  #gdf[col_name].mean() #or loubar, tp mean lbh oke kayanya
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # biar gak muncul centroid geographic crs shits
        centroidseries = gdf.centroid
    x, y = [list(t) for t in zip(*map(getXY,centroidseries))]
    matpos = np.array([x,y]).transpose()
    #d = cg.distance_matrix(matpos,threshold=1e6)
    print("calculating distance matrix..")
    d = calculate_distance_matrix(matpos)
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
    if len(loubar_dist_corr) > 0 and len(dist_corr) >0:
        eta = loubar_dist_corr.mean()  #/dist_corr.mean()
    else:
        eta = -1     
    return eta
#cellHP=cellHP[cellHP["Hval"] > 3 ]



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
    #name=city[city.ID_HDC_G0==cityID].eFUA_name
    #print("city id", (cityID,name.values[0]))
    geoser=city[city.index==cityID].geometry
    ####### intinya!!
    cellH=citywise_job(h, geoser, cityID, removefile=True, check_pixel=False)
    cellP=citywise_job(p, geoser, cityID, removefile=True, check_pixel=False)
    ######
    eqs = check_cell_diff(cellH,cellP)
    if eqs:
        #merging cellH and CellP = cellHP 
        cellHP=cellH.join(cellP.VALUE.rename("P"))
        cellHP.columns=["Hval", "geometry", "x", "y","Pval"]
        cellHP=cellHP[cellHP.Pval > 0 ] 
        cellHP=cellHP[cellHP.Hval > 0 ] #nan value entah knp jadi -2147483648 use H sebagai destinasi
        #cellHP.loc[cellHP["Pval"] < 0, "Pval"] = 0
        cellHP["3d_dens"] = cellHP["Pval"] / cellHP["Hval"]  #can be 0-div probelm,yields inf val ? == make gini and mse nan? hrs dihandle
        return cellHP
    else:
        print(cityID, "has different h and p cropped-raster data")
        return 0

def print_file_for_stats(cellHP, cityID, path="./data/cell_files_uc_vol"):
    ada = os.path.exists(path)
    if not ada :
       os.makedirs(path)
    cellHP[["Pval", "Hval", "3d_dens"]].to_csv(f"{path}/cell_"  + str(cityID) + ".csv")
    return 0

def print_file(cityID):
    save_path = "./data/cell_files_uc_vol/cell_"  + str(cityID) + ".gz"
    cellHP = get_cell(cityID)
    data = cellHP.to_dict()
    compressed_data = pickle.dumps(data)
    compressed_data = gzip.compress(compressed_data)
    with open(save_path, 'wb') as f:
        f.write(compressed_data)
    return 0

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


def calc_aggregate(cityID,   h_thrs = [15,25,35,45,55,65]):
    vars2exclude=["vars2exclude", "cellHP", "h_thrs","cbds" ]
    cellHP = get_cell(cityID)
    
    #cbds = [cellHP[cellHP["Hval"] >= h_thr] for h_thr in h_thrs ]
    #cbd_areas = [len(cbd) * 1e-2 for cbd in cbds] #in km2
    #get diff 
    #mse_pop_h = get_difference(cellHP,"Pval","Hval")
    #mse_2d_3d = get_difference(cellHP[cellHP["3d_dens"].notna()],"Pval","3d_dens")
    #jcr_pop_h  =  get_jcr_hot(cellHP, "Pval", "Hval")
    #jcr_2d_3d =  get_jcr_hot(cellHP[cellHP["3d_dens"].notna()], "Pval", "3d_dens")
    #gini_pop = get_gini(cellHP,"Pval")
    #gini_h = get_gini(cellHP,"Hval")
    #gini_3dpop = get_gini(cellHP[cellHP["3d_dens"].notna()], "3d_dens")

    #spr_pop = get_eta(cellHP,"Pval")
    #spr_h = get_eta(cellHP,"Hval")
    #spr_3dpop = get_eta(cellHP[cellHP["3d_dens"].notna()], "3d_dens") 

    avgpop3d,  avgh, totvol  = cellHP["3d_dens"].mean(), cellHP["Hval"].mean(), cellHP["Hval"].sum()
    #maxh, maxpop3d    = cellHP["3d_dens"].max(), cellHP["Hval"].max()
    
    local_vars=locals() # a dict of local variable in this function 
                        #(at this line so far)

    rets= {k: v for k, v in local_vars.items() if k not in vars2exclude}
    #make return variable (kecuali yang string namenya ada di vars2exclude)
    #next time gausah make col name "string" yg  mendokusai 
    #col name in dataframe of city_res bakal just as defined in this def !!!
    print_file_for_stats(cellHP, cityID)
    return list(rets.items()) # to temporary ngirit memory pas multiproses karena memori usage dict =  list * approx 4



def calc_aggr_large(cityID,njobs):
    #print(cityID)
    start = timeit.default_timer()
    vars2exclude=["vars2exclude", "cellHP", "h_thrs","cbds","cropped" ]
    cellHP = read_compressed("./data/cell_files_uc_vol/cell_"  + str(cityID) + ".gz")
    
    spr_pop = new_eta(cellHP,"Pval",njobs)
    spr_h = new_eta(cellHP,"Hval",njobs)

   # if len(cellHP[cellHP["3d_dens"].notna()]) > 0:
    #    spr_3dpop = get_eta(cellHP[cellHP["3d_dens"].notna()], "3d_dens") 
    #else:
    #    spr_3dpop = -1
    cropped = cellHP[cellHP["Hval"] > 3 ]

    if len(cropped) > 4:
        avgpop3d,  avgh  = cropped["3d_dens"].mean(), cropped["Hval"].mean()
        infpercap = (sum(cellHP["Hval"])*1e4) / sum(cellHP["Pval"])
    else:
        avgpop3d,avgh = -1,-1    
    #maxpop3d,  maxh  = cellHP["3d_dens"].max(), cellHP["Hval"].max()
    
    local_vars=locals() # a dict of local variable in this function 
                        #(at this line so far)

    rets= {k: v for k, v in local_vars.items() if k not in vars2exclude}
    #make return variable (kecuali yang string namenya ada di vars2exclude)
    #next time gausah make col name "string" yg  mendokusai 
    #col name in dataframe of city_res bakal just as defined in this def !!!
    stop= timeit.default_timer()
    print('Calculation time: ', stop - start)  
    return list(rets.items())

def get_quantity_by_radius(gdf,rads):
    vols = []
    pops = []
    centro = gdf.dissolve().centroid
    for r in rads:
        circ = centro.buffer(r)
        clipped = gdf.clip(circ)
        pops.append(clipped["Pval"].sum())
        vols.append(clipped["Hval"].sum())
    return vols, pops


def new_eta(gdf,col_name,njob):
    from scipy.spatial import distance
    gdf=gdf.reset_index(drop=True) 
    
    arbi=[(gdf["x"].values[0], gdf["y"].values[0])]
    denom=distance.cdist(arbi,  [(x, y) for x, y in zip(gdf.x, gdf.y)]).mean() 
    # asumsikan gdf== lattice, traversal of one2all kyknya mirip2 aja
    #-> one2all arbitari mewakili all2all dlm rata? & sama cepetnya sama radius calc

    thres =  gdf[col_name].mean()
    hotspot = gdf[gdf[col_name] >= thres]

    if len(hotspot)>0:
        hotspot_pos = np.array([hotspot["x"],hotspot["y"]]).transpose()
        hotspot_dist = calculate_avg_dist_chunk(hotspot_pos,njob)
        eta =  hotspot_dist/denom
        return eta
    else:
        return -1

def get_eta_barnes(gdf,col_name,theta):
    thres =  calc_loubar_threshold(gdf,col_name)  #gdf[col_name].mean() 
    hotspot = gdf[gdf[col_name] >= thres]
    centroidseries = hotspot.centroid
    x, y = [list(t) for t in zip(*map(getXY,centroidseries))]
    matpos = np.array([x,y]).transpose()
    hotspot_avg_dist = calc_avg_distance(matpos,theta)
    bound = gdf.dissolve().bounds
    radius = 0.5*np.sqrt(np.power((bound.loc[0].maxx - bound.loc[0].minx),2) + np.power((bound.loc[0].maxy - bound.loc[0].miny),2)) 
    avg_dist_all = (128*radius) / (45*np.pi)
    eta = hotspot_avg_dist #/ avg_dist_all
    return eta


# use this for pop < 5e6 ??
def calc_aggregate2(cityID, rads=[1000,2500,5000,7500,10000,12500,15000,17500,20000]):

    vars2exclude=["vars2exclude", "cellHP", "h_thrs","cbds","cropped" ,"rads"]
    #cellHP = gpd.read_file("./data/cell_files/cell_"  + str(cityID) + ".json")
    cellHP = read_compressed("./data/cell_files_uc_vol/cell_"  + str(cityID) + ".gz")
    #cbds = [cellHP[cellHP["Hval"] >= h_thr] for h_thr in h_thrs ]
    #cbd_areas = [len(cbd) * 1e-2 for cbd in cbds] #in km2
    #get diff 
  #  mse_pop_h = get_difference(cellHP,"Pval","Hval")
  #  mse_2d_3d = get_difference(cellHP[cellHP["3d_dens"].notna()],"Pval","3d_dens")
    #print(cellHP["Hval"])
  #  jcr_pop_h  = get_jcr_hot(cellHP, "Pval", "Hval")
  #  jcr_2d_3d =  get_jcr_hot(cellHP[cellHP["3d_dens"].notna()], "Pval", "3d_dens")
    from radial_scaling import radial_center_point

    a=radial_center_point( 50, 1000, cellHP)
    print(a)
    #gini_pop = get_gini(cellHP,"Pval")
    #gini_h = get_gini(cellHP,"Hval")
    #gini_3dpop = get_gini(cellHP[cellHP["3d_dens"].notna()], "3d_dens")
    #print(cityID)
    #if len(cellHP["Pval"]) > 4 and len(cellHP["Hval"]) > 4 : 
    spr_pop = get_eta_barnes(cellHP,"Pval",0.5)
    spr_pop_brute = get_eta(cellHP,"Pval")
    spr_h = get_eta_barnes(cellHP,"Hval",0.5)
    spr_dens = get_eta_barnes(cellHP,"3d_dens",0.5)
    #else:
    #    spr_pop = -1
    #    spr_h = -1    

   # if len(cellHP[cellHP["3d_dens"].notna()]) > 0:
    #    spr_3dpop = get_eta(cellHP[cellHP["3d_dens"].notna()], "3d_dens") 
    #else:
    #    spr_3dpop = -1
    #cropped = cellHP[cellHP["Hval"] > 3 ]
    #vols, pops = get_quantity_by_radius(cellHP,rads)

    #if len(cropped) > 4:
    avgpop3d,  avgh, totvol  = cellHP["3d_dens"].mean(), cellHP["Hval"].mean(), cellHP["Hval"].sum()

    infpercap =  cellHP["Hval"].sum() / cellHP["Pval"].sum()
    #else:
    #    avgpop3d,avgh = -1,-1    
    #maxpop3d,  maxh  = cellHP["3d_dens"].max(), cellHP["Hval"].max()
    
    local_vars=locals() # a dict of local variable in this function 
                        #(at this line so far)

    rets= {k: v for k, v in local_vars.items() if k not in vars2exclude}
    #make return variable (kecuali yang string namenya ada di vars2exclude)
    #next time gausah make col name "string" yg  mendokusai 
    #col name in dataframe of city_res bakal just as defined in this def !!!
    return list(rets.items()) # to temporary ngirit memory pas multiproses karena memori usage dict =  list * approx 4



#./data mesin koplo
ghsuc="./data/GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg"
#h="./data/GHS_BUILT_H_ANBH_E2018_GLOBE_R2022A_54009_100_V1_0/GHS_BUILT_H_ANBH_E2018_GLOBE_R2022A_54009_100_V1_0.tif"
h= "./data/GHS_BUILT_V_E2020_GLOBE_R2023A_54009_100_V1_0/GHS_BUILT_V_E2020_GLOBE_R2023A_54009_100_V1_0.tif"
p="./data/GHS_POP_E2020_GLOBE_R2022A_54009_100_V1_0/GHS_POP_E2020_GLOBE_R2022A_54009_100_V1_0.tif"

###./data mesin genta
#ghsuc="./data/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg"
#fua="./data/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
#h="./data/GHS_BUILT_V_E2020_GLOBE_R2022A_54009_100_V1_0.tif"
#p="./data/GHS_POP_E2015_GLOBE_R2022A_54009_100_V1_0.tif"


city = gpd.read_file(ghsuc).sort_values("P15",ascending=True)
#city = gpd.read_file(fua).sort_values("FUA_p_2015")
#print(len(city))

#city = city[city["P15"] < 1e6]
#print(len(city))
#city = city[city["P15"] >= 1e6]
city = city.head(5)
#city = city.iloc[1000:]
#city = city.sample(1).sort_values("P15")
#city=city.set_index("eFUA_ID", drop=False)
#cityID=city.index[-1]
city = city.set_index("ID_HDC_G0",drop=False)

#for running in cluster with SLURM
#num_cores= int(os.environ['SLURM_CPUS_PER_TASK'])
#mem = sklearn.get_config()['working_memory']
#print(mem)
#work_mem = (mem / (num_cores + 1))
num_cores=1
h_thrs=[15,25,35,45,55,65] #define h trsh first 

rads=[1000,2500,5000,7500,10000,12500,15000,17500,20000]  # in meter
#(biar bisa pake komprehensi list dan gak nulis2 lagi pas nge-wide list of areas)
print("Working with " +str(num_cores) + " cores for " + str(len(city)) + " cities")

def implementation_1():
    results = Parallel(n_jobs=num_cores, verbose=1)(delayed(calc_aggregate)(idx, h_thrs) for (idx) in city.ID_HDC_G0)
    city_res = city.join(pd.DataFrame([dict(d) for d in results]).set_index("cityID")) #blm tau run timenya kalo banyak 
    city_res = city_res.drop("ID_HDC_G0",axis=1)
    #city_res[["cbd_a_"+str(t) for t in h_thrs]] = pd.DataFrame(city_res.cbd_areas.to_list(), index=city_res.index)
    #city_res = city_res.drop("cbd_areas",axis=1)
    city_res.to_file(f"./data/hthr_{today}.json",driver="GeoJSON")

def implementation_2():
    Parallel(n_jobs=num_cores, verbose=1)(delayed(print_file)(idx) for (idx) in city.ID_HDC_G0)
    results = Parallel(n_jobs=num_cores, verbose=1)(delayed(calc_aggregate2)(idx, rads) for (idx) in city.ID_HDC_G0)
    city_res = city.join(pd.DataFrame([dict(d) for d in results]).set_index("cityID")) #blm tau run timenya kalo banyak 
    city_res = city_res.drop("ID_HDC_G0",axis=1)

    #city_res[["pop_"+str(t) for t in rads]] = pd.DataFrame(city_res.pops.to_list(), index=city_res.index)
    #city_res[["volume_"+str(t) for t in rads]] = pd.DataFrame(city_res.vols.to_list(), index=city_res.index)
    #city_res = city_res.drop("vols",axis=1)
    #city_res = city_res.drop("pops",axis=1)
    print(city_res["spr_pop"])
    print(city_res["spr_pop_brute"])
    city_res.to_file(f"./data/eta_builtvolume_{today}.json",driver="GeoJSON")

def implementation_large(num_cores):
    #print(len(city))
    results =[calc_aggr_large(idx,num_cores) for idx in tqdm(city.ID_HDC_G0)]
    city_res = city.join(pd.DataFrame([dict(d) for d in results]).set_index("cityID")) #blm tau run timenya kalo banyak 
    city_res = city_res.drop("ID_HDC_G0",axis=1)
    print(city_res)
    #city_res[["cbd_a_"+str(t) for t in h_thrs]] = pd.DataFrame(city_res.cbd_areas.to_list(), index=city_res.index)
    #city_res = city_res.drop("cbd_areas",axis=1)
    city_res.to_file(f"./data/avg3m_large_{today}.json",driver="GeoJSON")




def test():
    tracemalloc.start()
    start = timeit.default_timer()
    implementation_2()
    stop = timeit.default_timer()
    
    print('Ver 2 Time: ', stop - start)  
    print('Ver 2 Mem: ', tracemalloc.get_traced_memory())
    tracemalloc.stop()

def test_eta():
    import os
    r=[]
    new=[]
    old=[]
    for root, dirs, files in os.walk("./data/cell_files"):
         for name in files:
             if name.split(".")[-1]=="json":
                 r.append(os.path.join(root, name))
    for i in r:
        cellHP=gpd.read_file(i)
        cellHP=cellHP[cellHP["Hval"]>=3]
        if len(cellHP)>0:
            new.append(new_eta(cellHP,"Hval"))
            old.append(get_eta(cellHP, "Hval"))
    
    #plt.scatter(old, new[0:len(old)])
    #plt.xlabel("old")
    #plt.ylabel("new")
import radial_scaling as rs
from radial_scaling import radial_scaling_all 

if __name__ == '__main__':
    #a = [implement_quad(i) for i in city.ID_HDC_G0]
    #test()
    #implementation_2()
    #datas=rs.list_files("./data/cell_files_uc_vol/", "gz")
    #print(datas)
    #datas=[read_compressed(d) for d in datas]
    #radials = radial_scaling_all(datas, "Hval", start=10, num=1000)
    #radials.to_csv("./data/d_scaling.csv")

    #implementation_large(num_cores)
#cellHP=gpd.read_file("/Volumes/HDPH-UT/K-Jkt copy/cbd-slum/data/cell_files/cell_10523.0.json")
#cellHP=read_compressed("/Volumes/HDPH-UT/K-Jkt copy/cbd-slum/data/cell_files 2/cell_7165.0.gz")


#arbitary test
#path="/Volumes/HDPH-UT/K-Jkt copy/cbd-slum/data/cell_files 2/cell_2262.0.gz"
#cellHP=read_compressed(path)
#cellHP=cellHP[cellHP["Hval"]>3]
#start=timeit.default_timer()
#eta=new_eta(cellHP.sample(100), "Hval")
#stop=timeit.default_timer()
#print(f"runtime with {len(cellHP)} data ={stop-start}")



