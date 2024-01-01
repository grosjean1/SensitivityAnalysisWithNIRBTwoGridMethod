# -*- coding: utf-8 -*-
## Greedy Algorithm for NIRB
## Elise Grosjean
## 01/2021


from BasicTools.FE import FETools as FT
import numpy as np
from scipy import linalg

def orthoGS(basis_vectors,cptmodes,m,snapshotCorrelationOperator):
    residuals2=np.array(np.reshape(basis_vectors,((cptmodes,m))))
    basis_vectorsVec=np.reshape(basis_vectors,((cptmodes,m)))
            
    residuals2[-1,:]=basis_vectorsVec[-1]-sum((residuals2[k,:]*np.dot(snapshotCorrelationOperator.dot(residuals2[k]),basis_vectorsVec[-1]) for k in range(residuals2.shape[0]-1)))            
        
    basis_vectors[-1]=np.reshape(residuals2[-1,:],((m,1)))
    return basis_vectors

          
def orthogonality_check(Matrix,CorrelationMatrix,orthoN=False):
    """
    This function check for the pairwise orthogonality of the new basis
    """
    list_ = list(Matrix)
    dot_matrix = np.array([[np.dot(CorrelationMatrix.dot(item1), item2.T) for item1 in list_] for item2 in list_])
    A=dot_matrix

    if(orthoN==False):
        A=np.diag(np.diag(dot_matrix))
    else:
        A= np.eye(dot_matrix.shape[0])

    if (dot_matrix - A < 1e-12).all():
        print(True)
        return True
    else:
        error = dot_matrix - A
     
        print("max error with identity: ",np.max(np.abs(error)))
        return False


def greedy_algorithm(snapshots,NumerOfTimesSteps,snapshotCorrelationOperator, NumberOfModes):
    """
    Compute the greedy reduced basis using a greedy algorithm.
    """
   
    TolGreedy=1e-5 #Tolerance on the time steps
    DegreesOfFreedom=np.shape(snapshotCorrelationOperator)[0]
    NbParam=len(snapshots)
    
    # initialization: First parameter
    ind=0 #first parameter (orthonormalized, so norm(snapshot) > 1e-10)

    basis_vectors=[] # basis functions list
    snapshotsList=snapshots.copy()
    snapshotsArray=snapshotsList.pop(ind) #snapshots with the parameter ind...
    
    residuals=np.array(snapshotsArray) #...all time steps for first parameter 
    n,m = residuals.shape
            
    tol=1
    cptmodes=1 #First mode
    cpt=0
    
   
    tolold=tol
    matVecProduct0=snapshotCorrelationOperator.dot(residuals.T)

    # FIRST PARAMETER SELECTION //LOOP OVER TIME
    TimeIndices=[]
    #while(tol >= TolGreedy and NumberOfModes+1> cptmodes):
    GlobalIndex=[]

    while(2 >=  cptmodes and NumberOfModes+1> cptmodes):
    
        
        matVecProduct= snapshotCorrelationOperator.dot(residuals.T)
        NormTestMax=np.sqrt(np.diag(np.dot(residuals,matVecProduct)))
        TimeIndex=np.argmax(NormTestMax)
          
        mask = (TimeIndex not in TimeIndices)
        if mask==False: #check if not already in the time steps 
                
            TimeIndexOld=TimeIndex
            filtered_list = [elt for indx,elt in enumerate(list(np.arange(len(NormTestMax)))) if elt not in TimeIndices]
            TimeIndex = filtered_list[np.argmax(NormTestMax[filtered_list])] 
            print("New Time Index: ", TimeIndex)

        TimeIndices.append(TimeIndex)
        
        tol=NormTestMax[TimeIndex]

        basis_vectors.append(np.reshape(residuals[TimeIndex,:],((m,1)))/tol) # time index, added in the modes
        
        print(cptmodes," / ",NumberOfModes, "(same parameter, new time step)")
        print(NormTestMax[TimeIndex],"/",TolGreedy ,"(threshold)")
  
       
        residuals -= np.outer(basis_vectors[-1],np.dot(basis_vectors[-1].T,matVecProduct0)).T #update
        # Gram-Schmidt
        orthogonality=orthogonality_check(np.reshape(basis_vectors,((cptmodes,m))),snapshotCorrelationOperator)
        orthocpt=0
        
        while(10>=orthocpt and orthogonality==False):
            basis_vectors=orthoGS(basis_vectors,cptmodes,m,snapshotCorrelationOperator)
    
            orthogonality=orthogonality_check(np.reshape(basis_vectors,((cptmodes,m))),snapshotCorrelationOperator)
            orthocpt+=1

        cptmodes+=1
        cpt+=1
        """
        if(tol > tolold):
            print(tol)
            break;
        """
        tolold=tol
        print(cptmodes)
      
    GlobalIndex.append([ind,TimeIndices])    
    # PARAMETER SELECTION
    

  
    cptParam=1
    
    while(NumberOfModes>cptmodes):
        matrix=np.array(snapshotsList)    
        residuals = np.reshape(matrix,(((NbParam-cptParam)*n,m)))
        
        matVecProduct0 = snapshotCorrelationOperator.dot(residuals.T)
        for i in range(len(basis_vectors)):
            residuals -= np.outer(basis_vectors[i],np.dot(basis_vectors[i].T,matVecProduct0)).T
           
        matVecProduct = snapshotCorrelationOperator.dot(residuals.T)
        NormTestMax = np.sqrt(np.diag(np.dot(residuals,matVecProduct)))
        Index = np.argmax(NormTestMax)
        
        IndexParam = int(Index/n+0.00001)
        print("New param: ",IndexParam)
        tol =np.max(NormTestMax)
    
        basis_vectors.append(np.reshape(residuals[Index,:]/tol,((m,1)))) #Update
        
        print(cptmodes," / ",NumberOfModes, "(new parameter chosen)")
        print(tol,"/",TolGreedy,"(threshold)")
        
        if (cptParam!=1):
            cptmodes+=1

        #Gram-Schmidt
        orthogonality=orthogonality_check(np.reshape(basis_vectors,((cptmodes,m))),snapshotCorrelationOperator)
        orthocpt=0
        while(10>=orthocpt and orthogonality==False):
            basis_vectors=orthoGS(basis_vectors,cptmodes,m,snapshotCorrelationOperator)
          
            orthogonality=orthogonality_check(np.reshape(basis_vectors,((cptmodes,m))),snapshotCorrelationOperator)
            orthocpt+=1

            
        snapshotsArray=snapshotsList.pop(IndexParam)
    
        residuals=np.array(snapshotsArray)
        cpt=0
        #tol=1
        oldtol=tol
        TimeIndices=[]
        # TIME STEPS SELECTION
        TimeIndices.append(int(Index/m+0.00001))
        while(4>=cpt and NumberOfModes > cptmodes): #greedy on time steps
         
            if cpt==0:
                matVecProduct0=snapshotCorrelationOperator.dot(residuals.T)
                for i in range(len(basis_vectors)):
                    residuals -= np.outer(basis_vectors[i],np.dot(basis_vectors[i].T,matVecProduct0)).T
                   
            matVecProduct = snapshotCorrelationOperator.dot(residuals.T)
            NormTestMax = np.sqrt(np.diag(np.dot(residuals,matVecProduct)))
            TimeIndex = np.argmax(NormTestMax)
            mask = (TimeIndex not in TimeIndices)
            if mask==False: #check if not already in the time steps 
                
                TimeIndexOld=TimeIndex
                filtered_list = [elt for indx,elt in enumerate(list(np.arange(len(NormTestMax)))) if elt not in TimeIndices]
                TimeIndex = filtered_list[np.argmax(NormTestMax[filtered_list])]
                print("New Time Index: ", TimeIndex)
                #break
                
            print("time: ",TimeIndex)
            TimeIndices.append(TimeIndex)
            tol = np.max(NormTestMax)
            #if(tol >= oldtol-1e-8):
            #    break;
            basis_vectors.append(np.reshape(residuals[TimeIndex,:]/tol,((m,1))))
            cptmodes+=1
            print(cptmodes," / ",NumberOfModes, "(same parameter, new time step)")
            print(tol,"/",TolGreedy,"(threshold)")
            
            #Gram-Schmidt
            orthogonality=orthogonality_check(np.reshape(basis_vectors,((cptmodes,m))),snapshotCorrelationOperator)
            orthocpt=0
            
            while(10>=orthocpt and orthogonality==False):
                basis_vectors=orthoGS(basis_vectors,cptmodes,m,snapshotCorrelationOperator)
       
                orthogonality=orthogonality_check(np.reshape(basis_vectors,((cptmodes,m))),snapshotCorrelationOperator)
                orthocpt+=1
            
            residuals -= np.outer(basis_vectors[-1],np.dot(basis_vectors[-1].T,matVecProduct0)).T
            cpt+=1
            """
            if(tol >= tolold-1e-8):
                #cptParam+=1
                print(tol)
                break;
            """
            tolold=tol
            
            #if(tol >= oldtol-1e-8):
            #    break;
        cptParam+=1
        GlobalIndex.append([IndexParam,TimeIndices])  

    orthocpt=0
    orthogonality=orthogonality_check(np.reshape(basis_vectors,((cptmodes,m))),snapshotCorrelationOperator)
    while(10>=orthocpt and orthogonality==False):
        basis_vectors=orthoGS(basis_vectors,cptmodes,m,snapshotCorrelationOperator)
        
        orthogonality=orthogonality_check(np.reshape(basis_vectors,((cptmodes,m))),snapshotCorrelationOperator)
        orthocpt+=1


    reducedOrderBasisU=np.array(np.reshape(basis_vectors,((NumberOfModes,m))))
    
    cpt=0
    for i in range(NumberOfModes):
        reducedOrderBasisNorm=np.sqrt(reducedOrderBasisU[i,:]@(snapshotCorrelationOperator@reducedOrderBasisU[i,:]))
        reducedOrderBasisU[i,:]/=reducedOrderBasisNorm #np.sqrt(M[i,i]) #L2 orthonormalization


    
    ### H1 Orthogonalization
    
    #print(" H1 orthogonalization... ")
    RB=[]
    for k in range(NumberOfModes,NumberOfModes+1): #for k in range(1,NumberOfModes+1):
        """
        K=np.zeros((k,k)) #rigidity matrix
        M=np.zeros((k,k)) #mass matrix
        for i in range(k):
            matVecH1=h1ScalarProducMatrix.dot(reducedOrderBasisU[i,:])
            matVecL2=snapshotCorrelationOperator.dot(reducedOrderBasisU[i,:])
            for j in range(k):
                if i>=j:
            
                    K[i,j]=np.dot(matVecH1,reducedOrderBasisU[j,:])
                    M[i,j]=np.dot(matVecL2,reducedOrderBasisU[j,:])
                    K[j,i]=K[i,j]
                    M[j,i]=M[i,j]
    
    
        # Solving Kv=lambd Mv
        eigenValues,vr=linalg.eig(K, b=M) #eigenvalues + right eigenvectors
        print(eigenValues)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues#[idx]
    
        print("EV : ",eigenValues)
        print("sqrt(EV) : ",np.sqrt(eigenValues))
        print(np.shape(reducedOrderBasisU))
        
        reducedOrderBasisUp=reducedOrderBasisU[0:k,:]#[idx,:]
        
        eigenVectors = vr#[idx,:]
        
        reducedOrderBasisUp=np.dot(eigenVectors.transpose(),reducedOrderBasisUp)
        print(np.shape(eigenVectors))
    
    """
        reducedOrderBasisUp=reducedOrderBasisU
        orthogonality=orthogonality_check(reducedOrderBasisUp,snapshotCorrelationOperator,False)
        print("ortho l2?", orthogonality)
        #orthogonality=orthogonality_check(reducedOrderBasisUp,h1ScalarProducMatrix,False)
        #print("ortho h1?",orthogonality)
        reducedOrderBasisUt=reducedOrderBasisUp
    
        #for i in range(k):
        #    reducedOrderBasisUt[i]/=np.sqrt(eigenValues[i].real)
        RB.append(reducedOrderBasisUt)
    return RB,NumberOfModes,GlobalIndex









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




