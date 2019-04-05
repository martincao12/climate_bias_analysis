import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def removeNonsignificant(data):
    mean=np.mean(data)
    std=np.std(data)
    data=(data-mean)/std
    data.mask[(data<=np.percentile(data,95))&(data>=np.percentile(data,5))]=True
    return data

def grid2String(idxLat,idxLon):
    return "%d:%d"%(idxLat,idxLon)

def string2Grid(gridS):
    return [int(x) for x in gridS.split(":")]

def constructGraph(data,cyclic=True):
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

r=np.ma.array(np.load("data/diff_pr_MIROC5_GPCP.npy"),mask=np.load("data/diff_pr_MIROC5_GPCP_mask.npy"))
r=r[:,:,:240]
r=removeNonsignificant(r)[:,:,:]

r=r[::2,::2,0]



G=constructGraph(r,cyclic=False)


plt.grid(alpha=0.5)
plt.xticks([x for x in range(0,90)],[])
plt.yticks([x for x in range(0,45)],[])
plt.xlim(0,90)
plt.ylim(0,45)

components=nx.connected_component_subgraphs(G)
for component in components:
    # if len(component)<10:
    #     continue
    for node in component.nodes():
        y,x=string2Grid(node)
        if r[y,x]>0:
            plt.scatter([x+0.5],[y+0.5],s=50,c="r")
        else:
            plt.scatter([x+0.5],[y+0.5],s=50,c="b")
    

    for edge in component.edges():
        grid1,grid2=edge
        y1,x1=string2Grid(grid1)
        y2,x2=string2Grid(grid2)
        plt.plot([x1+0.5,x2+0.5],[y1+0.5,y2+0.5],c="g")



plt.show()