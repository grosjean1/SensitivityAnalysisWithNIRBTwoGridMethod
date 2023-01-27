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
onlineParameter=str(sys.argv[5])

## Directories
currentFolder=os.getcwd()
dataFolder=currentFolder
FinedataFolderU=osp.join(dataFolder,'FineSnapshots/'+sys.argv[2]) #for fine snapshots
CoarsedataFolderU=osp.join(dataFolder,'CoarseSnapshots/'+sys.argv[3]+'/'+sys.argv[4]) #for coarse snapshots
FinedataFolder=osp.join(dataFolder,'FineSnapshotsPhi/'+sys.argv[2]) #for fine snapshots
CoarsedataFolder=osp.join(dataFolder,'CoarseSnapshotsPhi/'+sys.argv[3]+'/'+sys.argv[4]) #for coarse snapshots
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
###  mesh space interpolation
------------------- 
"""
inputnodes=mesh2.nodes
outputnodes=mesh.nodes
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

print("param:",parameters)
parameters.remove(onlineParameter) #online parameter mu=1
print("parameters :",parameters)


snapshots=[]

for e,i in enumerate(parameters):
    snapshotsTime=[]
    for time in range(0,TF):   
        snapshot =MR.VTKReadToNp("Velocity",FinedataFolder+"/"+i+"/SnapshotPhih_",time)
        snapshotsTime.append(snapshot)
    snapshots.append(snapshotsTime)


snapshotsH=[]

for e,i in enumerate(parameters):
    snapshotsHTime=[]
    for time in range(0,TG):

        snapshotH =MR.VTKReadToNp("Velocity",CoarsedataFolder+"/"+i+"/SnapshotPhih_",time)
   
        #Compute the projected data using the projection operator
        snapshotHSpaceinterpolated = Operator.dot(snapshotH)
        snapshotsHTime.append(snapshotHSpaceinterpolated)
    print(i," ",len(oldtime),len(snapshotsHTime))
    interp  = interp1d(oldtime,snapshotsHTime,kind='quadratic',axis=0,fill_value="extrapolate")

    solutionUHI=interp(newtime) #time and space interpolation
    snapshotsH.append(solutionUHI)
  
    


############################################################
"""          Greedy                                      """
############################################################

print("ComputeL2ScalarProducMatrix ...")

l2ScalarProducMatrix = FT.ComputeL2ScalarProducMatrix( mesh, nbeOfComponentsPrimal)
#h1ScalarProducMatrix = FT.ComputeH10ScalarProductMatrix(mesh, nbeOfComponentsPrimal)

##### ALGO (full) GREEDY
reducedOrderBasisPhi,nev2,_=GD.Greedy(snapshots,TF,l2ScalarProducMatrix,h1ScalarProducMatrix=None,NumberOfModes=nev,Tol=1e-30)



print("Number of modes after greedyPhi",nev2)
############################################################
"""          Rectification  sur Phi                      """
############################################################

RI=np.zeros((TF,nev2,nev2)) #global rectification over the time steps
for time in range(TF):

    alpha=np.zeros((nev2,ns)) #fine coefficients
    beta=np.zeros((ns,nev2)) #coarse coefficients
    betainv=np.zeros((nev2,ns)) 
    R=np.zeros((nev2,nev2)) #rectification matrix
    
    for j,elt in enumerate(parameters):

        u1PT = snapshots[j][time]
        u1T = snapshotsH[j][time]
        
        for i in range(nev2):
            alpha[i,j]=u1PT@(l2ScalarProducMatrix@reducedOrderBasisPhi[i,:])
        for i in range(nev2):
            beta[j,i]=u1T@(l2ScalarProducMatrix@reducedOrderBasisPhi[i,:])
            betainv[i,j]=beta[j,i]
        
    lambd=1e-10 #Thikonov regularization parameter
    Rns=np.zeros((nev2,ns))
    
    for j in range(ns):
        Rns[:,j]=np.linalg.inv(beta.transpose()@beta+lambd*np.eye(nev2))@betainv[:,j]
        
    for i in range(nev2):
        R[i,:]=Rns@alpha[i,:]

    RI[time,:,:]=R


    
############################################################
"""          Save data for online part                   """
############################################################
### save reduced basis

outputNamePhi = "reducedOrderBasisPhi.pkl"
outputPhi = open(outputNamePhi, "wb")
pickle.dump(reducedOrderBasisPhi, outputPhi)
outputPhi.close()

OperatorOutput = "Operator.pkl"
outputOp=open(OperatorOutput,"wb")
pickle.dump(Operator, outputOp)
outputOp.close()

REGOutput = "Rectification.pkl"
outputReg=open(REGOutput,"wb")
pickle.dump(RI, outputReg)
outputReg.close()
