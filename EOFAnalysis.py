import numpy as np
import PlotHeatMap
from matplotlib import pyplot as plt

from eofs.standard import Eof

r=np.ma.array(np.load("data/diff_pr_CM5_GPCP.npy"),mask=np.load("data/diff_pr_CM5_GPCP_mask.npy")).transpose([2,0,1])
print(r.shape)
solver=Eof(r)
eofs=solver.eofs()
count=0
print(solver.varianceFraction())
print(solver.pcs()[:1])
plt.plot(np.arange(240),solver.pcs()[0])
plt.axhline(0)
plt.show()
for eof in eofs[:5]:
    count+=1
    # PlotHeatMap.plotGlobal(eof,"figure/CM5_bias_eof%d.png"%(count))