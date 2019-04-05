import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data=netCDF4.Dataset("data/raw_data/regrid/GPCP_regrid.nc")

pr=data.variables["pr"][12:252]/3600/24

data=netCDF4.Dataset("data/raw_data/regrid/pr_MIROC5_regrid.nc")
pr=data.variables["pr"][:240]


pr=pr[:240,:,:]*100000

x=np.arange(12)
y=np.arange(-45,46,2)
cmap=ListedColormap(("white","lightsteelblue","cornflowerblue","royalblue","darkblue","darkgreen","forestgreen","yellowgreen","greenyellow","yellow","orange","darkorange","orangered","red","darkred"))
monthMean=[]
for i in range(12):
    monthMean.append(np.mean(np.mean(pr[i::12,:,5:25],axis=0),axis=1)[22:68].data)

plt.contourf(x,y,np.array(monthMean).transpose([1,0]),np.arange(0,21,1),cmap="jet")
plt.colorbar()
plt.xticks(x,["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],fontsize=15)
plt.yticks(np.arange(-45,46,15),["45°S","30°S","15°S","0°","15°N","30°N","45°N"],fontsize=15)
plt.xlabel("Month",fontsize=20)
plt.ylabel("latitude",fontsize=20)
plt.tight_layout()
plt.savefig("figure/MIROC5_seasonal_cycle_CP.png")
plt.close()


monthMean=[]
for i in range(12):
    monthMean.append(np.mean(np.mean(pr[i::12,:,30:50],axis=0),axis=1)[22:68].data)

plt.contourf(x,y,np.array(monthMean).transpose([1,0]),np.arange(0,21,1),cmap="jet")
plt.colorbar()
plt.xticks(x,["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],fontsize=15)
plt.yticks(np.arange(-45,46,15),["45°S","30°S","15°S","0°","15°N","30°N","45°N"],fontsize=15)
plt.xlabel("Month",fontsize=20)
plt.ylabel("latitude",fontsize=20)
plt.tight_layout()
plt.savefig("figure/MIROC5_seasonal_cycle_EP.png")
plt.close()

