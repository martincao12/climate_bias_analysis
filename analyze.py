import numpy as np
import PlotHeatMap
import netCDF4
import networkx as nx
import time
from BiasInstance import BiasInstance
import mlpy
from BiasFamily import BiasFamily
import math
import matplotlib.pyplot as plt 
import os
import seaborn as sns
import matplotlib
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
from matplotlib import axes


figurePath=""
variableToAnalysis=""

deltaA=25
deltaR=0.5
deltaF=0.3
gammaR=2
gammaD=1
bfMinLen=5

def delFile(path):
    ls = os.listdir(path)
    for i in ls:
        cPath = os.path.join(path, i)
        if os.path.isdir(cPath):
            del_file(cPath)
        else:
            os.remove(cPath)

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

def removeNonsignificantOld(data):
    mean=np.mean(data)
    std=np.std(data)
    data=(data-mean)/std
    data.mask[(data<=np.percentile(data,97.5))&(data>=np.percentile(data,2.5))]=True
    return data

def removeNonsignificant(data,gammaR,normalize=False):
    dataPositive,dataNegative=dataPNSplit(data)
    meanPositive=np.mean(dataPositive)
    stdPositive=np.std(dataPositive)
    meanNegative=np.mean(dataNegative)
    stdNegative=np.std(dataNegative)
    data.mask[(data<=meanPositive+gammaR*stdPositive)&(data>=meanNegative-gammaR*stdNegative)]=True
    # data.mask[(data<=np.percentile(dataPositive,95))&(data>=np.percentile(dataNegative,5))]=True
    if normalize:
        data[data>0]=data[data>0]/meanPositive
        data[data<0]=data[data<0]/np.abs(meanNegative)
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
    # dist,_,(pathx,pathy)=mlpy.dtw_std(ts1,ts2,dist_only=False)
    # return dist/len(pathx)

def printStorage(s):
    for t in s:
        print(t)
        for i in range(len(s[t])):
            print(t,sorted(s[t][i].region))

def plotBiasFamily(bf,figName,dimLat=90,dimLon=180):
    bfHeatMap=np.zeros((dimLat,dimLon),dtype=np.int)
    bfTitle="Periods: "
    bfis=[]
    for bi in bf.bis:
        bfis.append(bi)
        for gridS in bi.region:
            idxLat,idxLon=string2Grid(gridS)
            bfHeatMap[idxLat,idxLon]=1
        # bfTitle+="%4d-%4d  "%(1980+bi.tStart,1980+bi.tEnd)
        bfTitle+="%04d%02d-%04d%02d  "%(1980+bi.tStart/12,bi.tStart%12+1,1980+bi.tEnd/12,bi.tEnd%12+1)
    PlotHeatMap.plotGlobal(bfHeatMap,figName,needBalance=True,title=bfTitle)
        

def identifyBiasInstances(data):
    s={}
    _,_,dimT=data.shape
    for t in range(dimT):
        print("extracting bias instances from time "+str(t))
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
        print("expanding bias instances start from time "+str(t))
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
    plotBIDistribution(bis)
    return bis

def obtainBiasFamilies(bis,gammaD):
    G=nx.Graph()
    G.add_nodes_from(range(len(bis)))

    DTWDistList=[]
    for i in range(len(bis)):
        for j in range(i+1,len(bis)):
            DTWDistList.append(normalizedDTWDistance(bis[i].center,bis[j].center))
    deltaF=np.mean(DTWDistList)-gammaD*np.std(DTWDistList)
    print("deltaF is set as",deltaF,np.mean(DTWDistList),np.std(DTWDistList))

    for i in range(len(bis)):
        for j in range(i+1,len(bis)):
            if normalizedDTWDistance(bis[i].center,bis[j].center)<=deltaF:
                if len(mergeGraph(bis[i].G,bis[j].G).nodes())>=deltaR*max(len(bis[i].region),len(bis[j].region)):
                    G.add_edge(i,j)
    components=nx.connected_components(G)

    bfs=[]
    for component in components:
        bfis=[]
        for idx in sorted(component):
            bi=bis[idx]
            bfis.append(bi)
        bf=BiasFamily(bfis)
        bf.updateCenter()
        bfs.append(bf)

    bfs=sorted(bfs,key=lambda bf:len(bf.center),reverse=True)

    plotBFDistribution(bfs)
    plotTop(bfs[:20])
    plotBFSel(bfs,[0,1,2])

    delFile(figurePath+"bis")
    delFile(figurePath+"bfs")


    count=0
    for bf in bfs:
        count+=1
        bfHeatMap=np.zeros((90,180),dtype=np.int)
        bfTitle="Periods: "
        bicount=0
        hist=[["Jan",0,0],["Feb",0,0],["Mar",0,0],["Apr",0,0],["May",0,0],["Jun",0,0],["Jul",0,0],["Aug",0,0],["Sep",0,0],["Oct",0,0],["Nov",0,0],["Dec",0,0]]
        trend=[["%04d%02d"%(1980+t/12,t%12+1),0] for t in range(240)]
        for bi in bf.bis:
            bicount+=1
            heatMap=np.zeros((90,180),dtype=np.int)
            for gridS in bi.region:
                idxLat,idxLon=string2Grid(gridS)
                heatMap[idxLat,idxLon]=1
                bfHeatMap[idxLat,idxLon]+=1
            # bfTitle+="%4d-%4d  "%(1980+bi.tStart,1980+bi.tEnd)
            bfTitle+="%04d%02d-%04d%02d  "%(1980+bi.tStart/12,bi.tStart%12+1,1980+bi.tEnd/12,bi.tEnd%12+1)
            for t in range(bi.tStart,bi.tEnd+1):
                if bi.center[t-bi.tStart]>0:
                    hist[t%12][1]+=1
                else:
                    hist[t%12][2]+=1
                trend[t][1]=bi.center[t-bi.tStart]
            # PlotHeatMap.plotGlobal(heatMap,"figure/bf%03d_%03d_%04d-%04d.png"%(count,bicount,1980+bi.tStart,1980+bi.tEnd),needBalance=True)
            PlotHeatMap.plotGlobal(heatMap,figurePath+"bis/bf%03d_%03d_%04d_%02d-%04d_%02d.png"%(count,bicount,1980+bi.tStart/12,bi.tStart%12+1,1980+bi.tEnd/12,bi.tEnd%12+1),needBalance=True)
        
        bfHeatMap=np.ma.array(bfHeatMap,mask=(bfHeatMap==0))
        bfHeatMap.mask[0,0]=False
        # bfHeatMap.mask[45,0]=False
        # if np.sum(bf.center)<0:
        #     bfHeatMap=-bfHeatMap
        PlotHeatMap.plotGlobal(bfHeatMap,figurePath+"bfs/bf%03d.png"%(count),needBalance=True,hist=hist,trend=trend)

    return bfs

def depictRelations(bfs):
    G=nx.Graph()
    G.add_nodes_from(range(len(bfs)))
    for i in range(len(bfs)):
        for j in range(i+1,len(bfs)):
            # print(1/math.exp(min(normalizedDTWDistance(bfs[i].center,bfs[j].center),normalizedDTWDistance(bfs[i].center,-bfs[j].center))))
            if len(bfs[i].center)<bfMinLen or len(bfs[j].center)<bfMinLen:
                continue
            similarity=1/math.exp(min(normalizedDTWDistance(bfs[i].center,bfs[j].center),normalizedDTWDistance(bfs[i].center,-bfs[j].center)))
            G.add_edge(i,j,similarity=similarity)

    edges=[(edge[0],edge[1]) for edge in sorted(G.edges(data=True),key=lambda edge:edge[2]["similarity"])]
    retain=1
    G.remove_edges_from(edges[:int(len(edges)*(1-retain))])

    
    delFile(figurePath+"relations")

    components=nx.connected_components(G)
    count=0
    for component in components:
        if len(component)<2:
            continue
        count+=1
        for idx in component:
            pass
            plotBiasFamily(bfs[idx],figName=figurePath+"relations/cluster%02d_bf%03d.png"%(count,idx))

    # pos=nx.spring_layout(G,k=0.5,iterations=50)
    # edgewidth=[]
    # for (u,v,d) in G.edges(data=True):
    #     edgewidth.append(round(G.get_edge_data(u,v)["similarity"]*5,2))
    # nx.draw_networkx_edges(G,pos,width=edgewidth)
    # nx.draw_networkx_nodes(G,pos,node_size=10)
    # plt.show()

def createDir():
    if not os.path.exists(figurePath):
        os.makedirs(figurePath)
    if not os.path.exists(figurePath+"bis"):
        os.makedirs(figurePath+"bis")
    if not os.path.exists(figurePath+"bfs"):
        os.makedirs(figurePath+"bfs")
    if not os.path.exists(figurePath+"relations"):
        os.makedirs(figurePath+"relations")

def plotBIDistribution(bis):
    lengths=[]
    for bi in bis:
        lengths.append(bi.tEnd-bi.tStart+1)
    distribution={}
    for length in lengths:
        if length not in distribution:
            distribution[length]=0
        distribution[length]+=1
    distribution=[[key,distribution[key]] for key in distribution]
    distribution=sorted(distribution,key=lambda x:x[0])
    # plt.bar([x[0] for x in distribution],[x[1] for x in distribution],log=True)
    # for x in distribution:
    #     plt.text(x[0]-0.09*len(str(x[1])),x[1]+0.5,str(x[1]),fontsize=15)
    # sns.countplot(data=distribution,log=True,edgecolor='k')
    sns.distplot(lengths,kde=False, hist_kws={'log':True,"edgecolor":'k'})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Lasting time(months)",fontsize=20)
    plt.ylabel("Number of bias instances",fontsize=20)
    plt.tight_layout()
    plt.savefig(figurePath+"bi_time_distribution.png")
    plt.close()

def plotBFDistribution(bfs):
    lengths=[]
    for bf in bfs:
        length=0
        for bi in bf.bis:
            length+=bi.tEnd-bi.tStart+1
        lengths.append(length)
    # distribution={}
    # for length in lengths:
    #     if length not in distribution:
    #         distribution[length]=0
    #     distribution[length]+=1
    # distribution=[[key,distribution[key]] for key in distribution]
    # distribution=sorted(distribution,key=lambda x:x[0])
    # plt.bar([x[0] for x in distribution],[x[1] for x in distribution],log=True)
    # for x in distribution:
    #     plt.text(x[0]-0.09*len(str(x[1])),x[1]+0.5,str(x[1]),fontsize=15)
    fig, ax = plt.subplots()
    # ax.set(yscale="log")
    sns.distplot(lengths,ax=ax,kde=False, hist_kws={'log':True,"edgecolor":'k'})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Lasting time(months)",fontsize=20)
    plt.ylabel("Number of bias families",fontsize=20)
    plt.tight_layout()
    plt.savefig(figurePath+"bf_time_distribution.png")
    plt.close()


    lengths=[]
    for bf in bfs:
        lengths.append(len(bf.bis))
    # distribution={}
    # for length in lengths:
    #     if length not in distribution:
    #         distribution[length]=0
    #     distribution[length]+=1
    # distribution=[[key,distribution[key]] for key in distribution]
    # distribution=sorted(distribution,key=lambda x:x[0])
    # plt.bar([x[0] for x in distribution],[x[1] for x in distribution],log=True)
    # for x in distribution:
    #     plt.text(x[0]-0.09*len(str(x[1])),x[1]+0.5,str(x[1]),fontsize=15)
    fig, ax = plt.subplots()
    # ax.set(yscale="log")
    sns.distplot(lengths,ax=ax,kde=False, hist_kws={'log':True,"edgecolor":'k'})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Number of bias instances contained",fontsize=20)
    plt.ylabel("Number of bias families",fontsize=20)
    plt.tight_layout()
    plt.savefig(figurePath+"bf_bi_distribution.png")
    plt.close()

def plotBFSel(bfs,sel):
    num=len(sel)
    fig=plt.figure(figsize=(18, num*2)) 
    gs = matplotlib.gridspec.GridSpec(num, 3,hspace=0.6,width_ratios=[5,5,8])
    for i in range(num):
        bf=bfs[sel[i]]
        heatMap=np.zeros((90,180))
        for bi in bf.bis:
            for gridS in bi.region:
                idxLat,idxLon=string2Grid(gridS)
                heatMap[idxLat,idxLon]+=1

        heatMap=np.ma.array(heatMap,mask=(heatMap==0))
        heatMap.mask[45,0]=False
        
        if np.sum(bf.center)<0:
            heatMap=-heatMap
        
        data=heatMap
        data=np.tile(data,[1,2])[:,int(data.shape[1]/2):int(data.shape[1]/2)+data.shape[1]]

        if np.abs(np.max(data))>np.abs(np.min(data)):
            data[45,0]=-np.abs(np.max(data))
        elif np.abs(np.max(data))<np.abs(np.min(data)):
            data[45,0]=np.abs(np.min(data))
        else:
            pass

        ax=plt.subplot(gs[i,0])
        m=Basemap(llcrnrlon=0,llcrnrlat=-45, urcrnrlon=360 , urcrnrlat=45,ax=ax)
        m.drawcoastlines(linewidth=0.5)


        parallels = np.arange(-90.,91.,30.)
        m.drawparallels(parallels,labels=[1,0,0,0],fontsize=12)
        meridians = np.arange(0.,361.,60.)
        m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=12)

        x=np.tile(np.arange(1,360,2),90).reshape(90,180)
        y=np.tile(np.arange(-89,90,2).reshape(90,1),180)


        cs=m.contourf(x,y,data,20,cmap = cm.seismic)
        # plt.text(300,-45,"No.%d"%(sel[i]+1),fontdict={"color":"black","weight":"bold","size":"15"})
        plt.title("No.%d"%(sel[i]+1),loc="left",fontdict={"color":"black","weight":"bold","size":"15"})


        hist=[["J",0,0],["F",0,0],["M",0,0],["A",0,0],["M",0,0],["J",0,0],["J",0,0],["A",0,0],["S",0,0],["O",0,0],["N",0,0],["D",0,0]]
        trend=[["%04d%02d"%(1980+t/12,t%12+1),0] for t in range(240)]
        for bi in bf.bis:
            for t in range(bi.tStart,bi.tEnd+1):
                if bi.center[t-bi.tStart]>0:
                    hist[t%12][1]+=1
                else:
                    hist[t%12][2]+=1
                trend[t][1]=bi.center[t-bi.tStart]
        ax=plt.subplot(gs[i,1])
        N=len(hist)
        index=np.arange(N)
        ax.bar(index-0.18,[x[1] for x in hist],facecolor="red",width=0.3,label="positive bias")
        ax.bar(index+0.18,[x[2] for x in hist],facecolor="blue",width=0.3,label="negative bias")
        # plt.xlabel("Months",fontsize=15)
        # plt.ylabel("Occurence",fontsize=15)
        plt.xticks(index,[x[0] for x in hist],fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc="upper left")
        # plt.text(9.8,0,"No.%d"%(sel[i]+1),fontdict={"color":"black","weight":"bold","size":"15"})





        ax3=plt.subplot(gs[i,2])
        ax3.plot([x[0] for x in trend],[x[1] for x in trend])

        totalLables=[x[0] for x in trend]
        xticks=list(range(0,len(trend),int(len(trend)/20)))
        xlabels=[totalLables[x] for x in xticks]
        # xticks.append(len(totalLables))
        # xlabels.append(totalLables[-1])
        ax3.set_xticks(xticks)
        ax3.set_xticklabels(xlabels, rotation=40,fontsize=12)
        plt.yticks(fontsize=12)


    plt.savefig(figurePath+"sel_bf.png")
    plt.close()


def plotTop(bfs):
    num=len(bfs)
    fig=plt.figure(figsize=(12, int((num+1)/2)*2)) 
    gs = matplotlib.gridspec.GridSpec(int((num+1)/2), 2,hspace=0)
    for i in range(num):
        bf=bfs[i]
        heatMap=np.zeros((90,180))
        for bi in bf.bis:
            for gridS in bi.region:
                idxLat,idxLon=string2Grid(gridS)
                heatMap[idxLat,idxLon]+=1

        heatMap=np.ma.array(heatMap,mask=(heatMap==0))
        heatMap.mask[45,0]=False
        
        if np.sum(bf.center)<0:
            heatMap=-heatMap
        
        data=heatMap
        data=np.tile(data,[1,2])[:,int(data.shape[1]/2):int(data.shape[1]/2)+data.shape[1]]

        if np.abs(np.max(data))>np.abs(np.min(data)):
            data[45,0]=-np.abs(np.max(data))
        elif np.abs(np.max(data))<np.abs(np.min(data)):
            data[45,0]=np.abs(np.min(data))
        else:
            pass

        ax=plt.subplot(gs[int(i/2),i%2])
        m=Basemap(llcrnrlon=0,llcrnrlat=-45, urcrnrlon=360 , urcrnrlat=45,ax=ax)
        m.drawcoastlines(linewidth=0.5)


        parallels = np.arange(-90.,91.,30.)
        m.drawparallels(parallels,labels=[1,0,0,0],fontsize=15)
        meridians = np.arange(0.,361.,60.)
        m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=15)

        x=np.tile(np.arange(1,360,2),90).reshape(90,180)
        y=np.tile(np.arange(-89,90,2).reshape(90,1),180)


        cs=m.contourf(x,y,data,20,cmap = cm.seismic)
        plt.text(300,-45,"No.%d"%(i+1),fontdict={"color":"black","weight":"bold","size":"15"})

    plt.savefig(figurePath+"top_bf.png")
    plt.close()



    fig=plt.figure(figsize=(12, int((num+1)/2)*1.8)) 
    gs = matplotlib.gridspec.GridSpec(int((num+1)/2), 2,hspace=0.2)
    for i in range(num):
        bf=bfs[i]
        hist=[["Jan",0,0],["Feb",0,0],["Mar",0,0],["Apr",0,0],["May",0,0],["Jun",0,0],["Jul",0,0],["Aug",0,0],["Sep",0,0],["Oct",0,0],["Nov",0,0],["Dec",0,0]]
        for bi in bf.bis:
            for t in range(bi.tStart,bi.tEnd+1):
                if bi.center[t-bi.tStart]>0:
                    hist[t%12][1]+=1
                else:
                    hist[t%12][2]+=1
        ax=plt.subplot(gs[int(i/2),i%2])
        N=len(hist)
        index=np.arange(N)
        ax.bar(index-0.18,[x[1] for x in hist],facecolor="red",width=0.3,label="positive bias")
        ax.bar(index+0.18,[x[2] for x in hist],facecolor="blue",width=0.3,label="negative bias")
        # plt.xlabel("Months",fontsize=15)
        # plt.ylabel("Occurence",fontsize=15)
        plt.xticks(index,[x[0] for x in hist],fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc="upper left")
        plt.text(9.8,0,"No.%d"%(i+1),fontdict={"color":"black","weight":"bold","size":"15"})
        # plt.title("Seasonal statistics",fontsize=25)
    # fig.tight_layout()
    plt.savefig(figurePath+"top_bf_distribution")
    plt.close()


if __name__ == '__main__':
    variableToAnalysis="diff_pr_CM5_GPCP"
    figurePath="figure/%s/"%(variableToAnalysis)
    createDir()

    start=time.time() 
    r=np.ma.array(np.load("data/%s.npy"%(variableToAnalysis)),mask=np.load("data/%s_mask.npy"%(variableToAnalysis)))*100000
    # r=loadData("data/diff_tos_ersst_ccsm4.txt")
    # for t in range(0,r.shape[2],12):
    #     r[:,:,int(t/12)]=np.mean(r[:,:,t:t+12],axis=2)
    # r=r[:,:,:int(r.shape[2]/12)]


    PlotHeatMap.plotGlobal(np.mean(r,axis=2),figurePath+"mean.png",needBalance=True)
    r=r[:,:,:240]
    r=removeNonsignificant(r,gammaR)[:,:,:]
    bis=identifyBiasInstances(r[:,:,:])
    print("all bias instances identified")
    print("%d bias instances identified"%(len(bis)))
    bfs=obtainBiasFamilies(bis,gammaD)
    print("bias families obtained")
    # depictRelations(bfs)
    # print("relation network contructed")
    print("total time cost: %d seconds"%(time.time()-start))

    






