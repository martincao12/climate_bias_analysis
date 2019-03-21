import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from mpl_toolkits.basemap import Basemap

def plotGlobal(data,figname,title="",needBalance=False,latStartAtZero=False,hist=[],trend=[]):
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

    fig=plt.figure(figsize=(20,12))        

    ax=fig.add_axes([0.1,0.1,0.8,0.7])

    if len(hist)!=0:
        gs = matplotlib.gridspec.GridSpec(2, 2, height_ratios=[9,3],width_ratios=[4,6]) 
        ax1=plt.subplot(gs[0,:])
        m=Basemap(llcrnrlon=0,llcrnrlat=-90, urcrnrlon=360 , urcrnrlat=90,ax=ax1)
    else:
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
    # cs=m.contourf(x,y,data,80,cmap = "gist_ncar")

    if len(hist)==0:
        cbar = m.colorbar(cs,location='bottom',pad="5%")
    titleW=""
    for i in range(0,len(title),140):
        titleW+=(title[i:i+140]+"\n")

    plt.title(titleW,fontsize="medium")


    if len(hist)!=0:
        ax2=plt.subplot(gs[1,0])
        N=len(hist)
        index=np.arange(N)
        ax2.bar(index-0.2,[x[1] for x in hist],facecolor="red",width=0.3,label="positive bias")
        ax2.bar(index+0.2,[x[2] for x in hist],facecolor="blue",width=0.3,label="negative bias")
        plt.xlabel("Months")
        plt.ylabel("occurence")
        plt.xticks(index,[x[0] for x in hist])
        plt.legend(loc="upper left")

        ax3=plt.subplot(gs[1,1])
        ax3.plot([x[0] for x in trend],[x[1] for x in trend])
        plt.xlabel("time")
        plt.ylabel("normalized bias")

        totalLables=[x[0] for x in trend]
        xticks=list(range(0,len(trend),int(len(trend)/20)))
        xlabels=[totalLables[x] for x in xticks]
        xticks.append(len(totalLables))
        xlabels.append(totalLables[-1])
        ax3.set_xticks(xticks)
        ax3.set_xticklabels(xlabels, rotation=40)


    plt.savefig(figname)
    plt.close()