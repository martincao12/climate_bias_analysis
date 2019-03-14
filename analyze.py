import numpy as np
import PlotHeatMap
import netCDF4
import networkx as nx
import time
from BiasInstance import BiasInstance
import mlpy

fileToAnalysis="data/diff_tos_ersst_ccsm3.txt"
# fileToAnalysis="data/diff_pr_gpcp_ccsm4.txt"
deltaA=20
deltaR=0.5
deltaF=1



def dataPNSplit(data):
    dataMask=data.mask
    dataValue=data.data
    dataMaskPositive=dataMask.copy()
    dataMaskPositive[dataValue<=0]=True
    dataMaskNegative=dataMask.copy()
    dataMaskNegative[dataValue>=0]=True
    dataPositive=np.ma.array(dataValue,mask=dataMaskPositive)
    dataNegative=np.ma.array(dataValue,mask=dataMaskNegative)
    return (dataPositive,dataNegative)

def removeNonsignificant(data,normalize=True):
    dataPositive,dataNegative=dataPNSplit(data)
    meanPositive=np.mean(dataPositive)
    stdPositive=np.std(dataPositive)
    meanNegative=np.mean(dataNegative)
    stdNegative=np.std(dataNegative)
    data.mask[(data<=meanPositive+stdPositive)&(data>=meanNegative-stdNegative)]=True
    data[data>0]=data[data>0]/meanPositive
    data[data<0]=data[data<0]/meanNegative
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

def findConnectedComponents(G):
    components=sorted(nx.connected_component_subgraphs(G),key=len,reverse=True)
    return components

def loadData(filename):
    input=open(filename,"r")
    dimLat,dimLon,dimT=[int(x) for x in input.readline().split(",")]
    r=np.zeros((dimLat,dimLon,dimT),dtype=np.float)
    rMask=np.ones((dimLat,dimLon,dimT),dtype=np.int)
    while True:
        line=input.readline()
        if not line:
            break
        lineArr=line.split(",")
        idxLat,idxLon,data=[int(lineArr[0])-1,int(lineArr[1])-1,[float(x) for x in lineArr[2:]]]
        r[idxLat,idxLon,:]=data
        rMask[idxLat,idxLon,:]=0
    r=-r #convert observation-simulation to simulation-observation
    #time start from 1980, latitude is -89 to 89 degrees north, longitude is -179 to 179 degress east
    return np.ma.array(r,mask=rMask)

def mergeGraph(G1,G2):
    G=nx.Graph()
    G.add_edges_from(set([tuple(sorted(edge)) for edge in G1.edges])&set([tuple(sorted(edge)) for edge in G2.edges]))
    connected_components=nx.connected_component_subgraphs(G)
    # if len(list(connected_components))==0:
    #     return nx.Graph()
    return max(connected_components,key=len,default=nx.Graph())

def computeBICenter(bi,data):
    bList=[]
    for gridS in bi.region:
        idxLat,idxLon=string2Grid(gridS)
        bList.append(data[idxLat,idxLon,bi.tStart:bi.tEnd+1])
    return np.mean(bList,axis=0)

def normalizedDTWDistance(ts1,ts2):
    return mlpy.dtw_std(ts1,ts2)/(len(ts1)+len(ts2))

def printStorage(s):
    for t in s:
        print(t)
        for i in range(len(s[t])):
            print(t,sorted(s[t][i].region))
        

def identifyBiasInstances(data):
    s={}
    _,_,dimT=data.shape
    for t in range(dimT):
        # PlotHeatMap.plotGlobal(data[:,:,t],"figure/data_%04d_%02d.png"%(1980+t/12,t%12+1),needBalance=True)
        G=constructGraph(data[:,:,t])
        components=findConnectedComponents(G)

        heatMap=np.zeros((data.shape[0],data.shape[1]),dtype=np.int)
        s[t]=[]
        for component in components:
            if len(component.nodes())<deltaA:
                continue
            s[t].append(BiasInstance(component.nodes(),t,t,component))
            for gridS in component.nodes():
                idxLat,idxLon=string2Grid(gridS)
                heatMap[idxLat,idxLon]=1
        # PlotHeatMap.plotGlobal(heatMap,"figure/component_%04d_%02d.png"%(1980+t/12,t%12+1),needBalance=True)

    for t in range(dimT):
        for i in range(len(s[t])):
            for tn in range(t+1,dimT):
                for j in range(len(s[tn])-1,-1,-1):
                    G=mergeGraph(s[t][i].G,s[tn][j].G)
                    if len(G.nodes())>=deltaA:
                        s[t][i].region=sorted(G.nodes())
                        s[t][i].G=G
                        s[t][i].tEnd=tn
                        del s[tn][j]
                        break
                if s[t][i].tEnd!=tn:
                    break
    
    bis=[]
    stat={}
    for key in s:
        for bi in s[key]:
            bi.center=computeBICenter(bi,data)
            bis.append(bi)
            heatMap=np.zeros((data.shape[0],data.shape[1]),dtype=np.int)
            for gridS in bi.region:
                idxLat,idxLon=string2Grid(gridS)
                heatMap[idxLat,idxLon]=1
            if bi.tEnd-bi.tStart+1 not in stat:
                stat[bi.tEnd-bi.tStart+1]=0
            stat[bi.tEnd-bi.tStart+1]+=1
            if bi.tEnd-bi.tStart+1>=1:
                pass
                # PlotHeatMap.plotGlobal(heatMap,"figure/component%03d_%04d-%04d.png"%(len(bis),1980+bi.tStart,1980+bi.tEnd),needBalance=True)
                # PlotHeatMap.plotGlobal(heatMap,"figure/component%03d_%04d_%02d-%04d_%02d.png"%(len(bis),1980+bi.tStart/12,bi.tStart%12+1,1980+bi.tEnd/12,bi.tEnd%12+1),needBalance=True)
    stat=sorted([[key,stat[key]] for key in stat],key=lambda x:x[0])
    return bis

def obtainBiasFamilies(bis):
    G=nx.Graph()
    G.add_nodes_from(range(len(bis)))
    for i in range(len(bis)):
        for j in range(i+1,len(bis)):
            if len(mergeGraph(bis[i].G,bis[j].G).nodes())>=deltaR*max(len(bis[i].region),len(bis[j].region)):
                if normalizedDTWDistance(bis[i].center,bis[j].center)<=deltaF or 1:
                    G.add_edge(i,j)
    components=nx.connected_components(G)
    count=0
    for component in components:
        count+=1
        bicount=0
        bfHeatMap=np.zeros((90,180),dtype=np.int)
        bfTitle="Periods: "
        for idx in sorted(component):
            bicount+=1
            bi=bis[idx]
            heatMap=np.zeros((90,180),dtype=np.int)
            for gridS in bi.region:
                idxLat,idxLon=string2Grid(gridS)
                heatMap[idxLat,idxLon]=1
                bfHeatMap[idxLat,idxLon]=1
            # bfTitle+="%4d-%4d  "%(1980+bi.tStart,1980+bi.tEnd)
            bfTitle+="%04d%02d-%04d%02d  "%(1980+bi.tStart/12,bi.tStart%12+1,1980+bi.tEnd/12,bi.tEnd%12+1)
            # PlotHeatMap.plotGlobal(heatMap,"figure/bf%03d_%03d_%04d-%04d.png"%(count,bicount,1980+bi.tStart,1980+bi.tEnd),needBalance=True)
            # PlotHeatMap.plotGlobal(heatMap,"figure/bf%03d_%03d_%04d_%02d-%04d_%02d.png"%(count,bicount,1980+bi.tStart/12,bi.tStart%12+1,1980+bi.tEnd/12,bi.tEnd%12+1),needBalance=True)
        PlotHeatMap.plotGlobal(bfHeatMap,"figure/bf%03d.png"%(count),needBalance=True,title=bfTitle)

if __name__ == '__main__':

    start=time.time() 
    r=loadData(fileToAnalysis)

    # for t in range(0,r.shape[2],12):
    #     r[:,:,int(t/12)]=np.mean(r[:,:,t:t+12],axis=2)
    # r=r[:,:,:int(r.shape[2]/12)]


    PlotHeatMap.plotGlobal(np.mean(r,axis=2),"figure/mean.png",needBalance=True)

    r=removeNonsignificant(r)
    bis=identifyBiasInstances(r[:,:,:])
    obtainBiasFamilies(bis)
    print("total time cost: %d seconds"%(time.time()-start))

    






