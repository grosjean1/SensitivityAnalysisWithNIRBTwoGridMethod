# -*- coding: utf-8 -*-
## Greedy Algorithm for NIRB
## Elise Grosjean
## 01/2021


from BasicTools.FE import FETools as FT
import numpy as np
from scipy import linalg

def orthogonality_check(Matrix,CorrelationMatrix): #to check if the RB is L2 orthogonal
    """
    This function check for the pairwise orthogonality of the new basis
    """
    list_ = list(Matrix)
    dot_matrix = np.array([[np.dot(CorrelationMatrix.dot(item1), item2) for item1 in list_] for item2 in list_])
    if (dot_matrix - np.eye(dot_matrix.shape[0]) < 1e-10).all():
        return True
    else:
        error = dot_matrix - np.eye(dot_matrix.shape[0])
        print("max error with identity: ",np.max(np.abs(error)))
        return False


def Greedy(snapshots,NumberOfTimeSteps,snapshotCorrelationOperator,h1ScalarProducMatrix=None,NumberOfModes=0,Tol=1e-12):
    """
    Greedy algorithm for the construction of the reduced basis
    orthogonal basis in H1 L2 or orthonormalized in L2
    #Algo as in https://arxiv.org/abs/2301.00761
    """
    
    TolTimeGreedy=1e-2 #tol for time compression 
    
    DegreesOfFreedom=np.shape(snapshotCorrelationOperator)[0]
    nbParam=len(snapshots)
   
    #first index
    ind=0 #normj.index(max(normj)) #first random parameter

    ListIndex=[ind] #first parameter
    norm=(np.sqrt(np.dot((snapshotCorrelationOperator.dot(snapshots[0][0])),snapshots[0][0])))
    if norm>1e-10:
        basis=[(snapshots[ind][0]/norm)] #first mode
    else:
        basis=[snapshos[ind][0]]

    TimeTol=1
    snapshotIterator1=snapshots[ind] #retrieve all time steps of first parameter
    print("first parameter index: ", ind)
    TimeIndex=[0]
    
    #  Greedy on time for the first parameter: time compression
    cptMode=0 #maximum iteration
    while(TimeTol>TolTimeGreedy and NumberOfModes>= cptMode):
         cptMode+=1
         TestVector=[[-1,-1]]*len(snapshotIterator1)
         for j in range(len(snapshotIterator1)):
             if not (j in TimeIndex): #if index not yet in the basis
                 w=snapshotIterator1[j]-sum((np.dot((snapshotCorrelationOperator.dot(b)),snapshotIterator1[j])*b for unused,b in enumerate(basis))) #Gram-Schmidt procedure
                 TestMax=np.sqrt(np.dot((snapshotCorrelationOperator.dot(w)),w))
                 TestVector[j]=[TestMax,w]
             else:
                 TestVector[j]=[-1,-1]
         NewIndex= TestVector.index(max(TestVector, key=lambda item: item[0])) #retrieve the time index of the maximum
         TimeTol=TestVector[NewIndex][0] #update in RB
         basis.append(TestVector[NewIndex][1]/TestVector[NewIndex][0]) #orthonormalization
         TimeIndex.append(NewIndex)
    print("First times: ", TimeIndex)

    GlobalIndex=[[ind,TimeIndex]]

    # Greedy on the parameters
    print("Time Tol",TimeTol, " / ",TolTimeGreedy)

    tol=1
    while(tol>Tol and NumberOfModes>=cptMode):
        cptMode+=1
        TestVector=[[-1,-1,-1]]*len(snapshots)
        for j in range(len(snapshots)):
             if not (j in ListIndex): #if index not yet in the basis
                  snapshotIterator1=snapshots[j] #retrieve all time steps
                  maxT=-1
                  for k in range(len(snapshotIterator1)):
                      w=snapshotIterator1[k]-sum((np.dot((snapshotCorrelationOperator.dot(b)),snapshotIterator1[k])*b for unused,b in enumerate(basis)))
                      TestMax=np.sqrt(np.dot((snapshotCorrelationOperator.dot(w)),w))
                      if TestMax>maxT:
                          TestVector[j]=[TestMax,w,k] #retrieve maximum value regarding the time steps
                          maxT=TestMax #update
             else:
                  TestVector[j]=[-1,-1,-1]
        NewIndex= TestVector.index(max(TestVector, key=lambda item: item[0])) #retrieve index of the maximum
        tol=TestVector[NewIndex][0]
        basis.append(TestVector[NewIndex][1]/TestVector[NewIndex][0])
        TimeIndex=TestVector[NewIndex][2]
        ListIndex.append(NewIndex)

        # Greedy on time for the first parameter

        snapshotIterator1=snapshots[NewIndex]
        TimeIndex=[TimeIndex]
        #print("tol t",tol, " ",tolGreedy , " " ,"cptMode",cptMode," ", NumberOfModes)
        # choose the time steps for this parameter
        print("new parameter: ", NewIndex)
        while(TimeTol>TolTimeGreedy and NumberOfModes>=cptMode):
            cptMode+=1
            TestVector=[[-1,-1]]*len(snapshotIterator1)
            for j in range(len(snapshotIterator1)):
                if not (j in TimeIndex): #if index not yet in the basis
                    w=snapshotIterator1[j]-sum((np.dot((snapshotCorrelationOperator.dot(b)),snapshotIterator1[j])*b for unused,b in enumerate(basis)))
                    TestMax=np.sqrt(np.dot((snapshotCorrelationOperator.dot(w)),w))
                    TestVector[j]=[TestMax,w]
                else:
                    TestVector[j]=[-1,-1]
            NewTimeIndex= TestVector.index(max(TestVector, key=lambda item: item[0])) #retrieve index of the maximum
            tol=TestVector[NewTimeIndex][0]
            basis.append(TestVector[NewTimeIndex][1]/TestVector[NewTimeIndex][0]) #orthonormalization
            TimeIndex.append(NewTimeIndex)
            print("times: ", TimeIndex)
        GlobalIndex.append([NewIndex,TimeIndex])
        
    NumberOfModes=min(NumberOfModes,len(basis))
    reducedOrderBasisU=np.zeros((NumberOfModes,DegreesOfFreedom)) #nev, Dof
    for i in range(NumberOfModes):
        reducedOrderBasisU[i,:]=basis[i]
    print("Number of modes: ", NumberOfModes)
    
    orthonogality=orthogonality_check(reducedOrderBasisU,snapshotCorrelationOperator)
    if orthonogality==False:       #Gram-schmidt 
          reducedOrderBasisU[0]=basis[0]
          
          for i in range(1,NumberOfModes):
             
              reducedOrderBasisU[i,:]=basis[i]-sum((reducedOrderBasisU[k,:]*np.dot(snapshotCorrelationOperator.dot(reducedOrderBasisU[k]),basis[i]) for k in range(i)))
              basis[i]=reducedOrderBasisU[i,:]
              reducedOrderBasisU[i]/=np.sqrt(np.dot((snapshotCorrelationOperator.dot(reducedOrderBasisU[i])),reducedOrderBasisU[i]))

    orthonogality=orthogonality_check(reducedOrderBasisU,snapshotCorrelationOperator)

    return reducedOrderBasisU,NumberOfModes,GlobalIndex




def GreedyNew(snapshots,NumberOfTimeSteps,snapshotCorrelationOperator,GlobalIndex,h1ScalarProducMatrix=None,NumberOfModes=0,Tol=1e-30): ##create a RB from a priori given parameters
    
    DegreesOfFreedom=np.shape(snapshotCorrelationOperator)[0]
    nbParam=len(snapshots)
   
    #first index
    ind=GlobalIndex[0][0] #index parameter
    indTime=GlobalIndex[0][1][0] #first time
    
    norm0=np.sqrt(np.dot(snapshotCorrelationOperator.dot(snapshots[ind][indTime]),snapshots[ind][indTime]))
    
    ListeIndex=[ind] #first parameter

    if norm0!=0:
        basis=[snapshots[ind][indTime]/norm0] #first basis function
    else:
        basis=[snapshots[ind][indTime]]
    snapshotIterator1=snapshots[ind]
    
    TimeIndex=[indTime]
    for k,telt in enumerate(GlobalIndex[0][1]):
        
        if k!=0:
            w=snapshotIterator1[telt]-sum((np.dot((snapshotCorrelationOperator.dot(b)),snapshotIterator1[telt])*b for unused,b in enumerate(basis)))
            norm0=np.sqrt(np.dot(snapshotCorrelationOperator.dot(w),w))
            if norm0!=0:
                basis.append(w/norm0) #first basis function
            else:
                basis.append(w)
            
    #  Greedy loop
    for i,elt in enumerate(GlobalIndex):
        if(i !=0 ):
            
            ind=GlobalIndex[i][0]
            snapshotIterator1=snapshots[ind] #retrieve all time steps for first parameter
            GlobalTime=GlobalIndex[i][1]
         
            for k,telt in enumerate(GlobalTime):
                w=snapshotIterator1[telt]-sum((np.dot((snapshotCorrelationOperator.dot(b)),snapshotIterator1[telt])*b for unused,b in enumerate(basis)))
                norm0=np.sqrt(np.dot(snapshotCorrelationOperator.dot(w),w))
                if norm0!=0:
                    basis.append(w/norm0) #first basis function
                else:
                    basis.append(w)
               
                #print("norm0: ",norm0)
                
    #print("nev new : ",NumberOfModes)
    reducedOrderBasisU=np.zeros((NumberOfModes,DegreesOfFreedom)) #nev, Dof
    for i in range(NumberOfModes):
        reducedOrderBasisU[i,:]=basis[i]
    print("Number of modes: ", NumberOfModes)
    
    orthonogality=orthogonality_check(reducedOrderBasisU,snapshotCorrelationOperator)
    if orthonogality==False:       #Gram-schmidt 
          reducedOrderBasisU[0]=basis[0]
          
          for i in range(1,NumberOfModes):
             
              reducedOrderBasisU[i,:]=basis[i]-sum((reducedOrderBasisU[k,:]*np.dot(snapshotCorrelationOperator.dot(reducedOrderBasisU[k]),basis[i]) for k in range(i)))
              basis[i]=reducedOrderBasisU[i,:]
              reducedOrderBasisU[i]/=np.sqrt(np.dot(snapshotCorrelationOperator.dot(reducedOrderBasisU[i]),reducedOrderBasisU[i]))

    orthonogality=orthogonality_check(reducedOrderBasisU,snapshotCorrelationOperator)
    return reducedOrderBasisU,NumberOfModes




