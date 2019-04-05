import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from mpl_toolkits.basemap import Basemap
import networkx as nx
from matplotlib.colors import ListedColormap

def removeNonsignificant(data):
    data.mask[(data<=np.percentile(data,90))&(data>=np.percentile(data,10))]=True
    return data

def grid2String(idxLat,idxLon):
    return "%d:%d"%(idxLat,idxLon)

def string2Grid(gridS):
    return [int(x) for x in gridS.split(":")]

def constructGraph(data,cyclic=False):
    dimLat,dimLon=data.shape
    G=nx.Graph()
    for i in range(dimLat):
        for j in range(dimLon):
            if data.mask[i,j]:
                continue
            if i+1<dimLat:
                if (not data.mask[i+1,j]) and ((data[i,j]>0 and data[i+1,j]>0) or (data[i,j]<0 and data[i+1,j]<0)):
                    G.add_edge(grid2String(i,j),grid2String(i+1,j))
            if j+1<dimLon:
                if (not data.mask[i,j+1]) and ((data[i,j]>0 and data[i,j+1]>0) or (data[i,j]<0 and data[i,j+1]<0)):
                    G.add_edge(grid2String(i,j),grid2String(i,j+1))
            if j+1==dimLon and cyclic:
                if (not data.mask[i,0]) and ((data[i,j]>0 and data[i,0]>0) or (data[i,j]<0 and data[i,0]<0)):
                    G.add_edge(grid2String(i,j),grid2String(i,0))
    return G

def mergeGraph(G1,G2):
    G=nx.Graph()
    G.add_edges_from(set([tuple(sorted(edge)) for edge in G1.edges])&set([tuple(sorted(edge)) for edge in G2.edges]))
    connected_components=nx.connected_component_subgraphs(G)
    # if len(list(connected_components))==0:
    #     return nx.Graph()
    return max(connected_components,key=len,default=nx.Graph())

def convertToHeatMap(G):
    heatMap=np.zeros((90,180))
    for node in G.nodes():
        lat,lon=string2Grid(node)
        heatMap[lat,lon]=1
    heatMap=np.ma.array(heatMap,mask=(heatMap==0))
    heatMap.mask[0,0]=False
    return heatMap


# r=np.ma.array(np.load("data/other_data/diff_tos_ersst_CCSM4.npy"),mask=np.load("data/other_data/diff_tos_ersst_CCSM4_mask.npy"))
r=np.ma.array(np.load("data/diff_pr_MIROC5_GPCP.npy"),mask=np.load("data/diff_pr_MIROC5_GPCP_mask.npy"))
r=r[:,:,:240]
r=removeNonsignificant(r)[:,:,:]

maxOverlap=0
triple=None


for t in range(235):
    print(t)
    components1=list(nx.connected_component_subgraphs(constructGraph(r[:,:,t])))
    components2=list(nx.connected_component_subgraphs(constructGraph(r[:,:,t+1])))


    for i in range(len(components1)):
        for j in range(len(components2)):
            G=mergeGraph(components1[i],components2[j])
            if len(G)<maxOverlap:
                continue
            if len(G)<max(len(components1[i]),len(components2[j]))*0.8:
                continue
            maxOverlap=len(G)
            triple=[components1[i],components2[j],G]
    


fig=plt.figure(figsize=(20,12))        

ax=fig.add_axes([0.1,0.1,0.8,0.7])


m=Basemap(llcrnrlon=0,llcrnrlat=-90, urcrnrlon=360 , urcrnrlat=90)
m.drawcoastlines()

# draw parallels.
parallels = np.arange(-90.,91.,30.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
meridians = np.arange(0.,361.,30.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)


x=np.tile(np.arange(1,360,2),90).reshape(90,180)
y=np.tile(np.arange(-89,90,2).reshape(90,1),180)

# data=np.random.randn(90,180)
# cs=m.contourf(x,y,data,20,cmap = cm.seismic)

# data=np.zeros((90,180))
# data[:10,:10]=1
cs=m.contourf(x,y,convertToHeatMap(triple[0])*1,2,cmap = ListedColormap(("white","blue",)),alpha=0.5)
cs=m.contourf(x,y,convertToHeatMap(triple[1])*2,2,cmap =  ListedColormap(("white","crimson")),alpha=0.5)
# data=convertToHeatMap(triple[0])+convertToHeatMap(triple[1])
# data.mask[data==1]=True
# cs=m.contour(x,y,data,2,linewidth=20,cmap = cm.tab10)
# cs=m.contourf(x,y,convertToHeatMap(triple[2])*2,3,cmap = ListedColormap(("white","blue","crimson")),alpha=1)



cbar = m.colorbar(cs,location='bottom',pad="5%")


plt.show()