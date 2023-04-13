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

#TolGreedy is for the time compression
def Greedy(snapshots,NumerOfTimesSteps,snapshotCorrelationOperator,NumberOfModes,h1ScalarProducMatrix=None, TolGreedy=1e-6):
    """
    Compute the greedy reduced basis using a greedy algorithm.
    """
   
    DegreesOfFreedom=np.shape(snapshotCorrelationOperator)[0]
    NbParam=len(snapshots)
    ListeParam=[k for k in range(NbParam)]
    
    # initialization: First parameter
    
    ind=0 #first parameter (orthonormalized, so norm(snapshot) > 1e-10)
    ListeIndex=[ind] 
    basis_vectors=[] # basis functions list
    snapshotsList=snapshots.copy()
    snapshotsArray=snapshotsList.pop(ind) 
    
    residuals=np.array(snapshotsArray) #all time steps for first parameter 
    n,m = residuals.shape
    
    tol=1
    cptmodes=1
    cpt=0
    matVecProduct0=np.zeros((m,n))
    NormTestMax0=np.zeros(m)
    #for k in range(1):
    GlobalIndex=[]
    while(tol >= TolGreedy and NumberOfModes+1> cptmodes):
        matVecProduct= snapshotCorrelationOperator.dot(residuals.T)
        NormTestMax=np.sqrt(np.diag(np.dot(residuals,matVecProduct)))
        
        if cpt==0:
            matVecProduct0=matVecProduct 
            NormTestMax0=NormTestMax
            TimeIndex=np.argmax(NormTestMax0)
            basis_vectors.append((np.reshape(residuals[TimeIndex,:],((m,1)))/NormTestMax[TimeIndex])) #first time index=random
            print(cptmodes," / ",NumberOfModes)
            print(NormTestMax[ind],"/",TolGreedy ,"(treshold)")
            ListeTimeIndices=[TimeIndex]
        else:
            TimeIndex= np.argmax(NormTestMax)
            tol=np.max(NormTestMax)
            print(cptmodes," / ",NumberOfModes, "(same parameter, new time step)")
            print(tol,"/",TolGreedy ,"(treshold)")

            basis_vectors.append(np.reshape(residuals[TimeIndex,:]/tol,((m,1))))
            ListeTimeIndices.append(TimeIndex)
        print("list",ListeTimeIndices)
        residuals -= np.outer(basis_vectors[-1],np.dot(basis_vectors[-1].T,matVecProduct0)).T
        cptmodes+=1
        cpt+=1
        
    GlobalIndex.append([ind,ListeTimeIndices])
    ListeParam.pop(ind)
    print(GlobalIndex)
    print("nb basis", len(basis_vectors))
    # greedy on the parameters

    matrix=np.array(snapshotsList)
    cptParam=1
    
    while(NumberOfModes+1>cptmodes):
       
        residuals = np.reshape(matrix,(((NbParam-cptParam)*n,m)))
        
        matVecProduct0 = snapshotCorrelationOperator.dot(residuals.T)
        for i in range(len(basis_vectors)):
            residuals -= np.outer(basis_vectors[i],np.dot(basis_vectors[i].T,matVecProduct0)).T
           
        matVecProduct = snapshotCorrelationOperator.dot(residuals.T)
        NormTestMax = np.sqrt(np.diag(np.dot(residuals,matVecProduct)))
        Index = np.argmax(NormTestMax)
        IndexParam = int(Index/n+0.00001)
        print("glob new index for new param: ", Index, "IndexParam: ", IndexParam, "TF : ", n)
        tol =np.max(NormTestMax)
    
        basis_vectors.append(np.reshape(residuals[Index,:]/tol,((m,1))))
        print(cptmodes," / ",NumberOfModes, "(new parameter)")
        print(tol,"/",TolGreedy,"(treshold)")
        cptmodes+=1
        snapshotsArray=snapshotsList.pop(IndexParam)
        matrix=np.array(snapshotsList)
        residuals=np.array(snapshotsArray)
        cpt=0
        tol=1
        ListeTimeIndices=[int(Index/(NbParam-cptParam)+0.00001)]

        while(tol >= TolGreedy and NumberOfModes >= cptmodes): #greedy on time steps
         
            if cpt==0:
                matVecProduct0=snapshotCorrelationOperator.dot(residuals.T)
                for i in range(len(basis_vectors)):
                    residuals -= np.outer(basis_vectors[i],np.dot(basis_vectors[i].T,matVecProduct0)).T
                   
            matVecProduct = snapshotCorrelationOperator.dot(residuals.T)
            NormTestMax = np.sqrt(np.diag(np.dot(residuals,matVecProduct)))
            TimeIndex = np.argmax(NormTestMax)
            ListeTimeIndices.append(TimeIndex)
            tol = np.max(NormTestMax)
            basis_vectors.append(np.reshape(residuals[TimeIndex,:]/tol,((m,1))))
            residuals -= np.outer(basis_vectors[-1],np.dot(basis_vectors[-1].T,matVecProduct0)).T
            print(cptmodes," / ",NumberOfModes, "(same parameter, new time step)")
            print(tol,"/",TolGreedy,"(treshold)")
            cptmodes+=1
            cpt+=1
            
            
            
        print(ListeTimeIndices, " ", IndexParam+cptParam)
        print(GlobalIndex)
        ind=ListeParam.pop(IndexParam)
        GlobalIndex.append([ind,ListeTimeIndices])
        cptParam+=1
    print(len(basis_vectors))
    basis_vectors=np.reshape(basis_vectors,((NumberOfModes,m)))
    reducedOrderBasisU=np.array(np.reshape(basis_vectors,((NumberOfModes,m))))
    
    orthogonality=orthogonality_check(reducedOrderBasisU,snapshotCorrelationOperator)
    cpt=0
    while (orthogonality==False and 10>=cpt):       #Gram-schmidt 
        for i in range(1,NumberOfModes):
              
            reducedOrderBasisU[i,:]=basis_vectors[i]-sum((reducedOrderBasisU[k,:]*np.dot(snapshotCorrelationOperator.dot(reducedOrderBasisU[k]),basis_vectors[i]) for k in range(i)))            
            reducedOrderBasisU[i]/=np.sqrt(np.dot((snapshotCorrelationOperator.dot(reducedOrderBasisU[i])),reducedOrderBasisU[i]))
        for j in range(1,NumberOfModes):
            basis_vectors[j]=reducedOrderBasisU[j,:]
        cpt+=1
        orthogonality=orthogonality_check(reducedOrderBasisU,snapshotCorrelationOperator)
        print(orthogonality)

       ### H1 Orthogonalization
    if(h1ScalarProducMatrix!=None):
        K=np.zeros((NumberOfModes,NumberOfModes)) #rigidity matrix
        M=np.zeros((NumberOfModes,NumberOfModes)) #mass matrix
        for i in range(NumberOfModes):
            matVecH1=h1ScalarProducMatrix.dot(reducedOrderBasisU[i,:])
            matVecL2=snapshotCorrelationOperator.dot(reducedOrderBasisU[i,:])
            for j in range(NumberOfModes):
                if i>=j:
            
                    K[i,j]=np.dot(matVecH1,reducedOrderBasisU[j,:])
                    M[i,j]=np.dot(matVecL2,reducedOrderBasisU[j,:])
                    K[j,i]=K[i,j]
                    M[j,i]=M[i,j]
    
    
        # on resoud Kv=lambd Mv
        eigenValues,vr=linalg.eig(K, b=M) #eigenvalues + right eigenvectors
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = vr[:, idx]
        reducedOrderBasisU=np.dot(eigenVectors.transpose(),reducedOrderBasisU)

        for i in range(NumberOfModes):
            reducedOrderBasisNorm=np.sqrt(reducedOrderBasisU[i,:]@(snapshotCorrelationOperator@reducedOrderBasisU[i,:]))
            reducedOrderBasisU[i,:]/=reducedOrderBasisNorm#np.sqrt(M[i,i]) #L2 orthonormalization
    
        orthogonality=orthogonality_check(reducedOrderBasisU,snapshotCorrelationOperator)
        print(orthogonality)
    return reducedOrderBasisU,NumberOfModes,GlobalIndex







def GreedyNew(snapshots,NumberOfTimeSteps,snapshotCorrelationOperator,GlobalIndex,h1ScalarProducMatrix=None,NumberOfModes=0,Tol=1e-30): ##create a RB from a priori given parameters
    
    DegreesOfFreedom=np.shape(snapshotCorrelationOperator)[0]
    nbParam=len(snapshots)
   
    #first index
    ind=GlobalIndex[0][0] #index parameter
    indTime=GlobalIndex[0][1][0] #first time
    
    norm0=np.sqrt(np.dot(snapshotCorrelationOperator.dot(snapshots[ind][indTime]),snapshots[ind][indTime]))
    
    ListeIndex=[ind] #first parameter

    if norm0>1e-10:
        basis=[snapshots[ind][indTime]/norm0] #first basis function
    else:
        basis=[snapshots[ind][indTime]]
    snapshotIterator1=snapshots[ind]
    
    TimeIndex=[indTime]
    for k,telt in enumerate(GlobalIndex[0][1]):
        
        if k!=0:
            w=snapshotIterator1[telt]-sum((np.dot((snapshotCorrelationOperator.dot(b)),snapshotIterator1[telt])*b for unused,b in enumerate(basis)))
            norm0=np.sqrt(np.dot(snapshotCorrelationOperator.dot(w),w))
            if norm0>=1e-10:
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




