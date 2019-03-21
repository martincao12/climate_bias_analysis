import os
import numpy as np
import PlotHeatMap
path="data/raw_data/domains/"

heatMap=np.zeros((90,180))
count=0
for filename in os.listdir(path):
    data=np.loadtxt(path+filename)
    count+=1
    heatMap[heatMap==0]+=(data*count)[heatMap==0]
heatMap=np.ma.array(heatMap,mask=(heatMap==0))
PlotHeatMap.plotGlobal(heatMap,"figure/CM5_bias_regions.png")