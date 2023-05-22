import os
import matplotlib.pyplot as plt
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import matplotlib

project='EPSG:4326'
df_path = "./hthr30-100.json"
city=gpd.read_file(df_path)
print(city.columns)
#norm=matplotlib.colors.LogNorm(vmin=0.001,vmax=100)

sc = plt.scatter(city["FUA_p_2015"],city["gini_pop"],c=city["cbd_a_50"],vmin=0,vmax=0.1, s=2,cmap="viridis")
plt.xscale("log")
plt.colorbar(sc)
plt.show()

#tolong dibikin lebih fancy wk