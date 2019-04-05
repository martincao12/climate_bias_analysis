import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def removeNonsignificant(data):
    mean=np.mean(data)
    std=np.std(data)
    data=(data-mean)/std
    data.mask[(data<=np.percentile(data,97.5))&(data>=np.percentile(data,2.5))]=True
    return data

def plotSlice():
    r=np.ma.array(np.load("data/diff_pr_MIROC5_GPCP.npy"),mask=np.load("data/diff_pr_MIROC5_GPCP_mask.npy"))
    r=r[:,:,:240]
    r=removeNonsignificant(r)[:,:,:]

    ax = plt.subplot(111, projection='3d') 

    data=[]

    for z in range(0,r.shape[2],1):
        slice_data=[]
        for x in range(r.shape[0]):
            for y in range(r.shape[1]):
                if r.mask[x,y,z]==False:
                    slice_data.append([x,y,z])
        data.append(slice_data)
    for slice_data in data[::60]:
        ax.scatter([x[0] for x in slice_data], [x[1] for x in slice_data], [x[2] for x in slice_data], c=randomcolor(),s=5)



    data=np.ones((r.shape[1],r.shape[0]))
    x=np.arange(r.shape[0])
    y=np.arange(r.shape[1])
    x,y=np.meshgrid(x,y)

    for height in range(0,r.shape[2],60):
        ax.plot_surface(x,y,data*height,alpha=0.2)




    ax.set_zlabel('time')  
    ax.set_ylabel('longitude')
    ax.set_xlabel('latitude')


    plt.show()


def plot2Slice():
    ax = plt.subplot(111, projection='3d')
    data=[]
    for x in range(30,61,2):
        for y in range(30,61,2):
            data.append([x,y,0])
    ax.scatter([x[0] for x in data],[x[1] for x in data],[x[2] for x in data],c="r")

    for x in range(35,65,2):
        for y in range(35,65,2):
            data.append([x,y,1])
    ax.scatter([x[0] for x in data],[x[1] for x in data],[x[2] for x in data],c="b")

    # ax.plot_trisurf([x[0] for x in data],[x[1] for x in data],[x[2] for x in data])

    for x in range(35,61):
        for y in range(35,61):
            ax.plot([x,x],[y,y],[0,1],c="b")
    ax.plot([35,35],[35,35],[0,1],c="r")
    ax.plot([35,35],[60,60],[0,1],c="r")
    ax.plot([60,60],[35,35],[0,1],c="r")
    ax.plot([60,60],[60,60],[0,1],c="r")

    ax.plot([35,35,60,60,35],[35,60,60,35,35],1,c="y")
    ax.plot([35,35,60,60,35],[35,60,60,35,35],0,c="y")



    data=np.ones((180,90))
    x=np.arange(90)
    y=np.arange(180)
    x,y=np.meshgrid(x,y)
    ax.plot_surface(x,y,data*0,alpha=0.2)
    ax.plot_surface(x,y,data*1,alpha=0.2)
    

    ax.set_xlim(0,90)
    ax.set_ylim(0,180)
    ax.set_zlim(0,2)
    ax.set_zticks([0,1])

    ax.set_zlabel('time') 
    ax.set_ylabel('longitude')
    ax.set_xlabel('latitude')


    plt.show()




def plotBI():
    ax = plt.subplot(111, projection='3d')
    
    
    data=[]
    for x in range(30,61,1):
        for y in range(30,61,1):
            data.append([x,y,0])

    ax.scatter([x[0] for x in data],[x[1] for x in data],[0 for x in data],c="b")
    ax.scatter([x[0] for x in data],[x[1] for x in data],[3 for x in data],c="b")
    ax.scatter([x[0] for x in data],[x[1] for x in data],[6 for x in data],c="b")
    ax.scatter([x[0] for x in data],[x[1] for x in data],[10 for x in data],c="b")
    ax.scatter([x[0] for x in data],[x[1] for x in data],[30 for x in data],c="b")
    ax.scatter([x[0] for x in data],[x[1] for x in data],[33 for x in data],c="b")

    lines=[]
    for i in range(30,60):
        lines.append([30,i])
        lines.append([60,i])
        lines.append([i,30])
        lines.append([i,60])

    lines=[]
    for x in range(30,61,1):
        for y in range(30,61,1):
            lines.append([x,y])

    for line in lines:
         ax.plot([line[0],line[0]],[line[1],line[1]],[0,3],c="b")
         ax.plot([line[0],line[0]],[line[1],line[1]],[6,10],c="b")
         ax.plot([line[0],line[0]],[line[1],line[1]],[30,33],c="b")


    ax.plot([30,30],[30,30],[0,3],c="r")
    ax.plot([30,30],[60,60],[0,3],c="r")
    ax.plot([60,60],[30,30],[0,3],c="r")
    ax.plot([60,60],[60,60],[0,3],c="r")
    ax.plot([30,30,60,60,30],[30,60,60,30,30],0,c="y")
    ax.plot([30,30,60,60,30],[30,60,60,30,30],3,c="y")


    ax.plot([30,30],[30,30],[6,10],c="r")
    ax.plot([30,30],[60,60],[6,10],c="r")
    ax.plot([60,60],[30,30],[6,10],c="r")
    ax.plot([60,60],[60,60],[6,10],c="r")
    ax.plot([30,30,60,60,30],[30,60,60,30,30],6,c="y")
    ax.plot([30,30,60,60,30],[30,60,60,30,30],10,c="y")


    ax.plot([30,30],[30,30],[30,33],c="r")
    ax.plot([30,30],[60,60],[30,33],c="r")
    ax.plot([60,60],[30,30],[30,33],c="r")
    ax.plot([60,60],[60,60],[30,33],c="r")
    ax.plot([30,30,60,60,30],[30,60,60,30,30],30,c="y")
    ax.plot([30,30,60,60,30],[30,60,60,30,30],33,c="y")




    data=[]
    for x in range(70,81,1):
        for y in range(120,151,1):
            data.append([x,y,0])
    ax.scatter([x[0] for x in data],[x[1] for x in data],[8 for x in data],c="b")
    ax.scatter([x[0] for x in data],[x[1] for x in data],[20 for x in data],c="b")

    lines=[]
    for x in range(70,81,1):
        for y in range(120,151,1):
            lines.append([x,y])

    for line in lines:
         ax.plot([line[0],line[0]],[line[1],line[1]],[8,20],c="b")


    ax.plot([70,70],[120,120],[8,20],c="r")
    ax.plot([70,70],[150,150],[8,20],c="r")
    ax.plot([80,80],[120,120],[8,20],c="r")
    ax.plot([80,80],[150,150],[8,20],c="r")
    ax.plot([70,70,80,80,70],[120,150,150,120,120],8,c="y")
    ax.plot([70,70,80,80,70],[120,150,150,120,120],20,c="y")




    #biasFamily
    ax.plot([30,30],[30,30],[3,6],c="g")
    ax.plot([30,30],[60,60],[3,6],c="g")
    ax.plot([60,60],[30,30],[3,6],c="g")
    ax.plot([60,60],[60,60],[3,6],c="g")

    ax.plot([30,30],[30,30],[10,30],c="g")
    ax.plot([30,30],[60,60],[10,30],c="g")
    ax.plot([60,60],[30,30],[10,30],c="g")
    ax.plot([60,60],[60,60],[10,30],c="g")





    ax.set_xlim(0,90)
    ax.set_ylim(0,180)
    ax.set_zlim(0,33)
    ax.set_zticks([0,3,6,8,10,20,30,33],[0,3,56,58,60,70,200,203])


    data=np.ones((180,90))
    x=np.arange(90)
    y=np.arange(180)
    x,y=np.meshgrid(x,y)
    for height in [0,3,6,8,10,20,30,33]:
        ax.plot_surface(x,y,data*height,alpha=0.2)


    



    ax.set_zlabel('time') 
    ax.set_ylabel('longitude')
    ax.set_xlabel('latitude')


    plt.show()



def plotBF():
    pass
    
plotSlice()