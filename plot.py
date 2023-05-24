import os
import matplotlib.pyplot as plt
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import matplotlib as mpl
#mpl.style.use('classic')
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
project='EPSG:4326'
df_path = "./hthr30-100.json"
city=gpd.read_file(df_path)
print(city.columns)
#norm=matplotlib.colors.LogNorm(vmin=0.001,vmax=100)
plt.figure()
plt.scatter(city["FUA_p_2015"],city["cbd_a_30"], s=5,c= "r", label=r"$H_{\rm min} = 30 \rm m$")
plt.scatter(city["FUA_p_2015"],city["cbd_a_40"], s=5,c="g", label=r"$H_{\rm min} = 40 \rm m$")
plt.scatter(city["FUA_p_2015"],city["cbd_a_50"], s=5,c= "b",label=r"$H_{\rm min} = 50 \rm m$")
plt.scatter(city["FUA_p_2015"],city["cbd_a_60"], s=5,c="c", label=r"$H_{\rm min} = 60 \rm m$")
plt.scatter(city["FUA_p_2015"],city["cbd_a_70"], s=5,c= "m",label=r"$H_{\rm min} = 70 \rm m$")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Population")
plt.ylabel(r"Area of high-rise building ($\rm km^2$)")
plt.legend()
plt.savefig("h-vs-pop-log.png")

plt.figure()
plt.scatter(city["FUA_p_2015"],city["cbd_a_30"], s=5,c= "r", label=r"$H_{\rm min} = 30 \rm m$")
plt.scatter(city["FUA_p_2015"],city["cbd_a_40"], s=5,c="g", label=r"$H_{\rm min} = 40 \rm m$")
plt.scatter(city["FUA_p_2015"],city["cbd_a_50"], s=5,c= "b",label=r"$H_{\rm min} = 50 \rm m$")
plt.scatter(city["FUA_p_2015"],city["cbd_a_60"], s=5,c="c", label=r"$H_{\rm min} = 60 \rm m$")
plt.scatter(city["FUA_p_2015"],city["cbd_a_70"], s=5,c= "m",label=r"$H_{\rm min} = 70 \rm m$")
plt.xscale("log")
#plt.yscale("log")
plt.ylim(-0.1,1)
plt.xlabel("Population")
plt.ylabel(r"Area of high-rise building ($\rm km^2$)")
plt.legend()
plt.savefig("h-vs-pop.png")

plt.figure()
sc = plt.scatter(city["gini_pop"],city["gini_h"], s=5,c=city["FUA_p_2015"], norm=mpl.colors.LogNorm(vmin=1e5,vmax=1e8))
plt.colorbar(sc)
#plt.xscale("log")
#plt.yscale("log")
#plt.ylim(-0.1,1)
plt.xlabel("Gini pop")
plt.ylabel("Gini H")
#plt.legend()
plt.savefig("ginih-vs-ginipop.png")

#tolong dibikin lebih fancy wk