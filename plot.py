import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib

project='EPSG:4326'
df_path = "./test.json"
city=gpd.read_file(df_path)
city["area_frac"] = city["cbd_areas"] / city["FUA_area"]

#norm=matplotlib.colors.LogNorm(vmin=0.001,vmax=100)

sc = plt.scatter(city["FUA_p_2015"],city["gini_pop"],c=city["cbd_areas"],vmin=0,vmax=1, s=2,cmap="viridis")
plt.xscale("log")
plt.colorbar(sc)
plt.show()

#tolong dibikin lebih fancy wk