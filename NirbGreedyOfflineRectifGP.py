# -*- coding: utf-8 -*-
## NIRB Sensivity test with OFFLINE/ONLINE DECOMPOSITION

## Elise Grosjean
## 01/2023

import numpy as np
import sys
import pickle

import os
import os.path as osp

import Readers as MR
import SolutionVTKWriter as SVTKW

from BasicTools.FE import FETools as FT

import Greedy as GD

from scipy import interpolate
from scipy.interpolate import interp1d #time interpolation

from scipy.spatial import cKDTree #space interpolation
from scipy.sparse import coo_matrix
        
############################################################
"""          Initialization                              """
############################################################

onlineParameter=str(sys.argv[5])

## Directories
dataFolder=os.getcwd()
FinedataFolderU=osp.join(dataFolder,'FineSnapshots/'+sys.argv[2]) #for fine snapshots
CoarsedataFolderU=osp.join(dataFolder,'CoarseSnapshots/'+sys.argv[3]+'/'+sys.argv[4]) #for coarse snapshots

FinedataFolder=osp.join(dataFolder,'FineSnapshotsPhi/'+sys.argv[2]) #for fine snapshots
CoarsedataFolder=osp.join(dataFolder,'CoarseSnapshotsPhi/'+sys.argv[3]+'/'+sys.argv[4]) #for coarse snapshots
print("Fine folder: ", FinedataFolder)
print("Coarse folder: ", CoarsedataFolder)

ns=0 #number of snapshots
count1=0
for _, folder, _ in os.walk(FinedataFolder): #number of folders in FineData
    count1 += len(folder)

ns=count1-1 #-1 because of the online parameter not included in offline snapshots
print("Number of snapshots: ",ns)
#ns=18

nev=int(sys.argv[1])   #modes number

dimension=2 #2D
           
TF=len([name for name in os.listdir(FinedataFolder+"/"+onlineParameter+"/")])
print("Number of fine time steps: ",TF)

TG=len([name for name in os.listdir(CoarsedataFolder+"/"+onlineParameter+"/")])
print("Number of coarse time steps: ",TG)

dtF=float(sys.argv[2]) #fine time steps
dtG=float(sys.argv[4]) #coarse time steps

TimeEnd=1.0001
for time in np.arange(0, TimeEnd, dtF):
    if time>=0:#.9999: #NIRB sur t0=]1, 2]
        t0f=time
        break
for time in np.arange(0, TimeEnd, dtG):
    if time>=0:#.9999:
        t0g=time
        break
    
#for time interpolation
oldtime=np.arange(t0g, TimeEnd, dtG)
newtime=np.arange(t0f, TimeEnd, dtF)

"""
-------------------
###  Read fine mesh
------------------- 
"""

meshFileName = FinedataFolder + "/"+onlineParameter+"/SnapshotPhih_1.vtu";
FineMesh=MR.Readmesh(meshFileName)
FineMesh.nodes= FineMesh.nodes[:,:2] #2D

print("Fine mesh defined in " + meshFileName + " has been read")
nbeOfComponentsPrimal = 1 # 1 field 
numberOfNodes = FineMesh.GetNumberOfNodes()
print("DoF fine mesh ", numberOfNodes)

"""
-------------------
###  Read coarse mesh
------------------- 
"""
meshFileName2 = CoarsedataFolder + "/"+onlineParameter+"/SnapshotPhih_1.vtu";#//FineMesh/mesh1.msh"
CoarseMesh=MR.Readmesh(meshFileName2)
CoarseMesh.nodes = CoarseMesh.nodes[:,:2] #CAS 2D

print("Coarse mesh defined in " + meshFileName2 + " has been read")
numberOfNodes2 = CoarseMesh.GetNumberOfNodes()
print("DoF coarse mesh ", numberOfNodes2)


"""
-------------------
###  mesh space interpolation: nearest # other interpolation may be used
------------------- 
"""
inputnodes=CoarseMesh.nodes
outputnodes=FineMesh.nodes
kdt = cKDTree(inputnodes)
nbtp = outputnodes.shape[0]
_, ids = kdt.query(outputnodes)
cols=ids
row = np.arange(nbtp)
data = np.ones(nbtp)
Operator=coo_matrix((data, (row, cols)), shape=(nbtp , inputnodes.shape[0]))
        
"""
-------------------
###  read all snapshots ...
------------------- 
"""

parameters=[]
for i in range(1,ns+2):
    if(float(0.5*i)%1>1e-3):
        parameters.append(str(0.5*i))
    else:
        parameters.append(str(int(0.5*i)))

parameters.remove(onlineParameter) # online parameter mu=1
print("parameters: ",parameters)

""" Fine snapshots for Psi """

snapshots=[]

for e,i in enumerate(parameters):
    snapshotsTime=[]
    for time in range(0,TF):   
        snapshot =MR.VTKReadToNp("Velocity",FinedataFolder+"/"+i+"/SnapshotPhih_",time)
        snapshotsTime.append(snapshot)
    snapshots.append(snapshotsTime)


""" Fine snapshots for U """

snapshotsU=[]

for e,i in enumerate(parameters):
    snapshotsTime=[]
    for time in range(0,TF):   
        snapshot =MR.VTKReadToNp("Velocity",FinedataFolderU+"/"+i+"/Snapshoth_",time)
        snapshotsTime.append(snapshot)
    snapshotsU.append(snapshotsTime)


""" Coarse snapshots for U """

snapshotsHU=[]

for e,i in enumerate(parameters):
    snapshotsHTime=[]
    for time in range(0,TG):

        snapshotH =MR.VTKReadToNp("Velocity",CoarsedataFolderU+"/"+i+"/Snapshoth_",time)
   
        #Compute the projected data using the projection operator
        snapshotHSpaceinterpolated = Operator.dot(snapshotH)
        snapshotsHTime.append(snapshotHSpaceinterpolated)
    
    interp  = interp1d(oldtime,snapshotsHTime,kind='quadratic',axis=0,fill_value="extrapolate")

    solutionUHI=interp(newtime) #time and space interpolation
    snapshotsHU.append(solutionUHI)
  

############################################################
"""         Offline: Greedy                                      """
############################################################

print("Compute Mass Matrix ...")

l2ScalarProducMatrix = FT.ComputeL2ScalarProducMatrix( FineMesh, nbeOfComponentsPrimal)
#h1ScalarProducMatrix = FT.ComputeH10ScalarProductMatrix(FineMesh, nbeOfComponentsPrimal)

##### ALGO GREEDY
reducedOrderBasisU,nev1,indices1=GD.Greedy(snapshotsU,TF,l2ScalarProducMatrix, nev)
reducedOrderBasisPhi,nev2=GD.GreedyNew(snapshots,TF,l2ScalarProducMatrix,indices1,NumberOfModes=nev1) #if suitable parameters for the second RB 

print("1:",indices1)
#print("2:",indices2)
#print("Number of modes after greedyPhi",nev)
print("Number of modes after greedyU",nev1)

############################################################
"""          Rectification                               """
############################################################
# define the function
#import tensorflow as tf

from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, WhiteKernel,ConstantKernel


alpha=np.zeros((nev1*TF,ns)) #fine coefficients
alphainv=np.zeros((ns,nev1*TF)) #fine coefficients

beta=np.zeros((ns,nev2*TF)) #coarse coefficients

#kernel = 1.0 * RBF(length_scale=1., length_scale_bounds=(1e-1, 1e0))  + WhiteKernel(noise_level=1, noise_level_bounds=(1e-8, 1e-1))
kernel=ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed") # + WhiteKernel(noise_level=1, noise_level_bounds=(1e-8, 1e-5)) #kernel for GP regression 
for time in range(TF):
    
    for j,elt in enumerate(parameters):

        u1PT = snapshots[j][time]
        u1T = snapshotsHU[j][time]
        
        for i in range(nev1):
            alpha[i*TF+time,j]=u1PT@(l2ScalarProducMatrix@reducedOrderBasisPhi[i,:])
            alphainv[j,i*TF+time]=alpha[i*TF+time,j]
    
            beta[j,i*TF+time]=u1T@(l2ScalarProducMatrix@reducedOrderBasisU[i,:])

X = beta
y = alphainv
"""
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(X,y,c="yellow")
"""
REG3 = GaussianProcessRegressor(kernel=kernel).fit(X, y) #GP 


    
############################################################
"""          Save data for online part                   """
############################################################
### save reduced basis
outputNameU = "reducedOrderBasisU.pkl"
outputU = open(outputNameU, "wb")
pickle.dump(reducedOrderBasisU, outputU)
outputU.close()

outputNamePhi = "reducedOrderBasisPhi.pkl"
outputPhi = open(outputNamePhi, "wb")
pickle.dump(reducedOrderBasisPhi, outputPhi)
outputPhi.close()

OperatorOutput = "Operator.pkl"
outputOp=open(OperatorOutput,"wb")
pickle.dump(Operator, outputOp)
outputOp.close()

REGOutput = "Regression.pkl"
outputReg=open(REGOutput,"wb")
pickle.dump(REG3, outputReg)
outputReg.close()
