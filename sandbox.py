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
import pandas as pd
import numpy as np
from datetime import date
today = date.today().strftime("%d_%m_%Y")
pd.options.mode.chained_assignment = None 
import matplotlib.pyplot as plt

ghsuc="./data/GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg"
h="./data/GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100_V1_0/GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100_V1_0.tif"
p="./data/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0.tif"

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
 
def check_cell_diff(cell1,cell2):
    xdiff=((cell1.x-cell2.x)==0).all()
    ydiff=((cell1.y-cell2.y)==0).all()
    xymatch=pd.Series([xdiff, ydiff]).all()
    lenmatch= len(cell1)==len(cell2)
    equals = xymatch and lenmatch
    return equals
 
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
        cellHP=cellHP[cellHP.Hval>=0] #nan value entah knp jadi -2147483648 use H sebagai destinasi
        cellHP.loc[cellHP["Pval"] <0, "Pval"] = 0
        cellHP["3d_dens"] = cellHP["Pval"] / cellHP["Hval"]  #can be 0-div probelm,yields inf val ? == make gini and mse nan? hrs dihandle
        return cellHP
    else:
        print(cityID, "has different h and p cropped-raster data")
        return 0
 
city = gpd.read_file(ghsuc).sort_values("P15")
print(len(city))

large_city = city[city["P15"] >= 5e6]
print(len(large_city))

city_list = city[["ID_HDC_G0", "UC_NM_MN"]]
large_city_list = large_city[["ID_HDC_G0", "UC_NM_MN"]]
    
jakarta_uc = get_cell(11861)

sample_city = city[city["UC_NM_MN"] == "Bangkok"]
sample_city = sample_city[["ID_HDC_G0", "UC_NM_MN"]]

fig, ax = plt.subplots()
jakarta_uc.plot(ax=ax)

jakarta_uc.plot()

jakarta = city[city["UC_NM_MN"] == "Jakarta"]
    
jakarta_uc[["Hval", "Pval", "3d_dens"]]

jakarta_uc.sort_values(by=['3d_dens'])


jakarta_df = jakarta_uc.loc[:, jakarta_uc.columns!='geometry']

jakarta_lowrise = jakarta_df[jakarta_df["Hval"] <= 5]
jakarta_lowrise = jakarta_lowrise[jakarta_lowrise["Hval"] > 0]


highest_h = jakarta_uc[jakarta_uc["Hval"] > 20]

f, ax = plt.subplots(1, figsize=(6, 6))
# Base layer with all the areas for the background
for poly in jakarta_uc['geometry']:
    gpd.plotting.plot_multipolygon(ax, poly, facecolor='black', linewidth=0.025)
# Smallest areas
for poly in highest_h['geometry']:
    gpd.plotting.plot_multipolygon(ax, poly, alpha=1, facecolor='red', linewidth=0)
ax.set_axis_off()
f.suptitle('Areas with highest h')
plt.axis('equal')
plt.show()

jakarta_uc.plot(column='Pval', legend=True)
jakarta_uc.plot(column='Hval', legend=True)
jakarta_uc.plot(column='3d_dens', legend=True, cmap='OrRd')

jakarta_uc.plot(column="Pval", legend=True, scheme="quantiles", figsize=(15, 10), missing_kwds={"color": "lightgrey", "label": "Missing values", })
jakarta_uc.plot(column="3d_dens", legend=True, scheme="quantiles", figsize=(15, 10), missing_kwds={"color": "lightgrey", "label": "Missing values", })

jakarta_3d_dens = jakarta_uc.loc[:, jakarta_uc.columns=='3d_dens']
jakarta_low_3d_dens = jakarta_uc[jakarta_uc["3d_dens"] <= 50]
jakarta_low_3d_dens.plot(column='3d_dens', legend=True)

plt.hist(jakarta_3d_dens)
jakarta_high_3d_dens = jakarta_uc[jakarta_uc["3d_dens"] > 30]
jakarta_high_3d_dens.plot(column='3d_dens', legend=True)

jakarta_Pval = jakarta_uc.loc[:, jakarta_uc.columns=='Pval']
plt.hist(jakarta_Pval)
jakarta_low_Pval = jakarta_uc[jakarta_uc["Pval"] <= 600]
jakarta_low_Pval.plot(column='Pval', legend=True)

jakarta_high_Pval = jakarta_uc[jakarta_uc["Pval"] > 200]
jakarta_high_Pval.plot(column='Pval', legend=True)

berlin_uc = get_cell(2850)
berlin_uc.plot(column='Hval', legend=True)
berlin_uc.plot(column='3d_dens', legend=True)

bangkok_uc = get_cell(10714)
bangkok_uc.plot(column='Hval', legend=True)

plt.savefig('berlin.png', dpi=1080)



from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Extract the relevant features from your dataset (e.g., 'x', 'y', and '3d_dens')
X = jakarta_high_3d_dens[['x', 'y', '3d_dens']]

# Create a DBSCAN instance with appropriate parameters
# eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
# min_samples: The number of samples (data points) in a neighborhood for a point to be considered as a core point.
dbscan = DBSCAN(eps=100, min_samples=10)

# Fit DBSCAN to your data
dbscan.fit(X)

# Extract the cluster labels (-1 represents noise points)
labels = dbscan.labels_

# Visualize the clusters
plt.scatter(X['x'], X['y'], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.colorbar(label='Cluster Label')
plt.show()