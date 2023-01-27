
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

     
############################################################
"""          Online part                                 """
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

inputName="Operator.pkl"
Operator=pickle.load(open(inputName, "rb"))


"""
-------------------
###  RB
------------------- 
"""


inputName="reducedOrderBasisU.pkl"
reducedOrderBasisU=pickle.load(open(inputName, "rb"))
nev1=np.shape(reducedOrderBasisU)[0]

inputName="reducedOrderBasisPhi.pkl"
reducedOrderBasisPhi=pickle.load(open(inputName, "rb"))
nev2=np.shape(reducedOrderBasisPhi)[0]

inputName="Regression.pkl"
REG3=pickle.load(open(inputName, "rb"))


print("Compute Mass Matrix ...")

l2ScalarProducMatrix = FT.ComputeL2ScalarProducMatrix( FineMesh, nbeOfComponentsPrimal)
#h1ScalarProducMatrix = FT.ComputeH10ScalarProductMatrix(FineMesh, nbeOfComponentsPrimal)


#######################################################################

for i in [onlineParameter]:
    
    snapshotsHTime=[]
    for time in range(TG):
        print(time)
        snapshotH =MR.VTKReadToNp("Velocity",CoarsedataFolderU+"/"+i+"/Snapshoth_",time)
    
        #Compute the projected data using the projection operator
        snapshotHSpaceinterpolated = Operator.dot(snapshotH)
        snapshotsHTime.append(snapshotHSpaceinterpolated)
    
    
    interp  = interp1d(oldtime,snapshotsHTime,kind='quadratic',axis=0,fill_value="extrapolate")
    solutionUHI=interp(newtime) #time and space interpolation
    CompressedSolutionUj=np.zeros(nev2*TF)
    for time in range(TF):

        u1PT=solutionUHI[time]
        for j in range(nev2):
                CompressedSolutionUj[j*TF+time]=u1PT@(l2ScalarProducMatrix@reducedOrderBasisU[j,:])
    y_1=REG3.predict(CompressedSolutionUj.reshape(1,nev2*TF)) #predict with GP regression
    
    """   
    plt.scatter(CompressedSolutionUj,y_1,c="red")
    print(nev2)
    """
    y_1=np.array(y_1[0]).reshape((nev2,TF))
    
    for time in range(TF):
      
        reconstructedCompressedSolution = np.dot(y_1[:,time], reducedOrderBasisPhi) 
#  
  
        ##################################################
        #######   saving solution in VTK ############

        VTKBase = MR.VTKReadmesh(meshFileName)
        SVTKW.numpyToVTKWrite(VTKBase,reconstructedCompressedSolution,"NIRB_approximation_"+str(time)+"_"+str(nev)+".vtu")


    ##################################################

