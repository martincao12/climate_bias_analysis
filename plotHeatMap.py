import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from mpl_toolkits.basemap import Basemap

def plotGlobal(data,figname,title="",needBalance=False,latStartAtZero=False):
    # data[0:5,0:5]=1

    if not latStartAtZero:
        data=np.tile(data,[1,2])[:,int(data.shape[1]/2):int(data.shape[1]/2)+data.shape[1]]

    if needBalance:
        if np.abs(np.max(data))>np.abs(np.min(data)):
            data[0,0]=-np.abs(np.max(data))
        elif np.abs(np.max(data))<np.abs(np.min(data)):
            data[0,0]=np.abs(np.min(data))
        else:
            pass

    fig=plt.figure(figsize=(16,8))
    ax=fig.add_axes([0.1,0.1,0.8,0.7])

    m=Basemap(llcrnrlon=0,llcrnrlat=-90, urcrnrlon=360 , urcrnrlat=90)
    m.drawcoastlines()

    # draw parallels.
    parallels = np.arange(-90.,91.,30.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    # draw meridians
    meridians = np.arange(0.,361.,30.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

    # nx=data.shape[1]
    # ny=data.shape[0]
    # lons,lats=m.makegrid(nx,ny)
    # x,y=m(lons,lats)


    x=np.tile(np.arange(1,360,2),90).reshape(90,180)
    y=np.tile(np.arange(-89,90,2).reshape(90,1),180)


    cs=m.contourf(x,y,data,20,cmap = cm.seismic)
    cbar = m.colorbar(cs,location='bottom',pad="5%")
    plt.title(title,fontsize="medium")
    plt.savefig(figname)
    plt.close()