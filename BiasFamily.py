import numpy as np
class BiasFamily:
    def __init__(self,bis):
        self.bis=bis
        self.center=None
    
    def updateCenter(self):
        self.bis=sorted(self.bis,key=lambda bi:bi.tStart)
        self.center=np.array([])
        for bi in self.bis:
            self.center=np.append(self.center,bi.center)