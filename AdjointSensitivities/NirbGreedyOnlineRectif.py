# -*- coding: utf-8 -*-
## NIRB parabolic test with OFFLINE/ONLINE DECOMPOSITION

## Elise Grosjean
## 01/2022



import numpy as np
import sys
import pickle
import os
import os.path as osp

import Readers as MR
import SolutionVTKWriter as SVTKW

from BasicTools.FE import FETools as FT
#import pickle

import Greedy as GD
#import SVD

from scipy import linalg
from scipy import interpolate
#from scipy.interpolate import griddata 
from scipy.interpolate import interp1d #time interpolation

from scipy.spatial import cKDTree #space interpolation
from scipy.sparse import coo_matrix
        
############################################################
"""          Initialization                              """
############################################################
print("dans la rectification!")
onlineParameter= str(sys.argv[5])
## Directories
currentFolder=os.getcwd()
dataFolder=currentFolder

FinedataFolder=osp.join(dataFolder,'FineAdjoint/'+sys.argv[2]) #for fine snapshots
CoarsedataFolder=osp.join(dataFolder,'CoarseAdjoint/'+sys.argv[3]+'/'+sys.argv[4]) #for coarse snapshots
print("fine folder: ", FinedataFolder)
print("coarse folder: ", CoarsedataFolder)


ns=0 #number of snapshots
count1=0
for _, folder, _ in os.walk(FinedataFolder): #number of folders in FineData
    count1 += len(folder)

ns=count1-1 #-1 because of the online parameter not included in offline snapshots
print("Number of snapshots: ",ns)
#ns=18


nev=int(sys.argv[1])   #nombre de modes

time=0.0 #init
dimension=2 #2D
           
TF=len([name for name in os.listdir(FinedataFolder+"/1/")])
print("Number of fine time steps: ",TF)

TG=len([name for name in os.listdir(CoarsedataFolder+"/1/")])
print("Number of coarse time steps: ",TG)

dtF=float(sys.argv[2]) #fine time steps
dtG=float(sys.argv[4]) #coarse time steps

for time in np.arange(0, 1.0001, dtF):
    if time>=0:#.9999: #NIRB sur t0=]1, 2]
        t0f=time
        break
for time in np.arange(0, 1.0001, dtG):
    if time>=0:#.9999:
        t0g=time
        break
    
#for time interpolation
oldtime=np.arange(t0g, 1.0001, dtG)
newtime=np.arange(t0f, 1.0001, dtF)

"""
-------------------
###  Read fine mesh
------------------- 
"""

meshFileName = FinedataFolder + "/1/SnapshotPhih_1.vtu";
mesh=MR.Readmesh(meshFileName)
mesh.nodes= mesh.nodes[:,:2] #2D

print("Fine mesh defined in " + meshFileName + " has been read")
nbeOfComponentsPrimal = 1 # 1 field 
numberOfNodes = mesh.GetNumberOfNodes()
print("DoF fine mesh ", numberOfNodes)
"""
-------------------
###  Read coarse mesh
------------------- 
"""
meshFileName2 = CoarsedataFolder + "/1/SnapshotPhih_1.vtu";#//FineMesh/mesh1.msh"
mesh2=MR.Readmesh(meshFileName2)
mesh2.nodes = mesh2.nodes[:,:2] #CAS 2D

print("Coarse mesh defined in " + meshFileName2 + " has been read")
numberOfNodes2 = mesh2.GetNumberOfNodes()
print("DoF coarse mesh ", numberOfNodes2)


"""
-------------------
###  mesh space interpolation: nearest # other interpolation may be used
------------------- 
"""

inputName="Operator.pkl"
Operator=pickle.load(open(inputName, "rb"))


"""
-------------------
###  RB
------------------- 
"""

inputName="reducedOrderBasisPhi.pkl"
reducedOrderBasisPhi=pickle.load(open(inputName, "rb"))
nev2=np.shape(reducedOrderBasisPhi)[0]

inputName="Rectification.pkl"
RI=pickle.load(open(inputName, "rb"))

print("Compute Mass Matrix ...")

l2ScalarProducMatrix = FT.ComputeL2ScalarProducMatrix( mesh, nbeOfComponentsPrimal)
#h1ScalarProducMatrix = FT.ComputeH10ScalarProductMatrix(FineMesh, nbeOfComponentsPrimal)


############################################################
"""          Online part                                 """
############################################################

for i in [onlineParameter]:

    snapshotsHTime=[]
    for time in range(TG):
        snapshotH =MR.VTKReadToNp("Velocity",CoarsedataFolder+"/"+i+"/SnapshotPhih_",time)
   
        #Compute the projected data using the projection operator
        snapshotHSpaceinterpolated = Operator.dot(snapshotH)
        snapshotsHTime.append(snapshotHSpaceinterpolated)
   
    interp  = interp1d(oldtime,snapshotsHTime,kind='quadratic',axis=0,fill_value="extrapolate")
    solutionUHI=interp(newtime) #time and space interpolation
    print(np.shape(solutionUHI[0]))

    for time in range(TF):    
        R=RI[time,:,:] #nev1 nev2
       
        u1PT=solutionUHI[time]
        coef=np.zeros(nev2)
        CompressedSolutionUj=np.zeros(nev2)
        for j in range(nev2):
            CompressedSolutionUj[j]=u1PT@(l2ScalarProducMatrix@reducedOrderBasisPhi[j,:])

        for i in range(nev2):
            coef[i]=0
            for j in range(nev2):
                coef[i]+=R[i,j]*CompressedSolutionUj[j]

        reconstructedCompressedSolution = np.dot(coef, reducedOrderBasisPhi) #rectified nirb
        #reconstructedCompressedSolution = np.dot(CompressedSolutionUj, reducedOrderBasisPhi) #classical nirb without rectification
  
        ##################################################
        #######   saving solution in VTK ############

        VTKBase = MR.VTKReadmesh(meshFileName)
        SVTKW.numpyToVTKWrite(VTKBase,reconstructedCompressedSolution,"NIRB_approximation_"+str(time)+"_"+str(nev)+".vtu")
        ##################################################

        
