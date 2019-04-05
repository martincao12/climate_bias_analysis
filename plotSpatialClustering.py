import os
import numpy as np
import PlotHeatMap
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from mpl_toolkits.basemap import Basemap
import networkx as nx
from matplotlib.colors import ListedColormap
import random

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    # return "#"+color
    return ["r","g","b","brown"][np.random.randint(0,4)]
    # return "b"

path="data/raw_data/domains/"

heatMap=np.zeros((90,180))
count=0
for filename in os.listdir(path):
    data=np.loadtxt(path+filename)
    count+=1
    heatMap[heatMap==0]+=(data*count)[heatMap==0]
heatMap=np.ma.array(heatMap,mask=(heatMap==0))
PlotHeatMap.plotGlobal(heatMap,"figure/MIROC5_bias_regions.png")



fig=plt.figure(figsize=(20,12))        

ax=fig.add_axes([0.1,0.1,0.8,0.7])


m=Basemap(llcrnrlon=0,llcrnrlat=-90, urcrnrlon=360 , urcrnrlat=90)
m.drawcoastlines()

parallels = np.arange(-90.,91.,30.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=20)

meridians = np.arange(0.,361.,60.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=20)


x=np.tile(np.arange(1,360,2),90).reshape(90,180)
y=np.tile(np.arange(-89,90,2).reshape(90,1),180)

heatMap=np.zeros((90,180))
cr=0
for filename in os.listdir(path):
    heatMap=np.zeros((90,180))
    heatMap=np.loadtxt(path+filename)
    heatMap=np.ma.array(heatMap,mask=(heatMap==0))
    heatMap.mask[0,0]=False
    heatMap.data[0,0]=0
    cr=cr%20
    cs=m.contourf(x,y,heatMap,2,cmap = ListedColormap(("white",randomcolor())),alpha=0.5)
    cr+=1

# cbar = m.colorbar(cs,location='bottom',pad="5%")

plt.show()