#! /bin/bash

#Elise Grosjean
#lance NIRB + POD Greedy Parabolique avec mu=1, a comparer avec erreur FEM fine, ex: nev=10 (H=h*2 et h=H²) finetime=0.01 0.02 0.05 0.1
# uses :
#           - script  NirbGreedyTestNIRBOKParabolique.py with mu=1
#           - meshio-convert (vtk) to convert vtu files to vtk...

finetime="0.01 0.02 0.05 0.1"
finemesh="140 70 30 15"
coarsemesh="15 10 7 5"

#grossier
#coarsetimediv="0.02 0.04 0.1 0.2" #hdiv2
coarsetimesqrt="0.1 0.1414 0.22 0.32" #sqrth

for nev in 10 #a priori number of modes
do
    values=1 #first fine size mesh
    for Param in 1 ################# value of parameter in [1:0.5:9]
    do
	
	tau=$(echo $finetime | cut -d ' ' -f $values) # time step dtF
	echo time $tau 
	
	## for sqrth ##
	taucoarse=$(echo $coarsetimesqrt | cut -d ' ' -f $values) #time step dtG

       	python3	NirbGreedyOfflineRectif.py $nev $tau sqrth $taucoarse $Param #launch offline+online parts ## Deterministic Rectification
	python3	NirbGreedyOnlineRectif.py $nev $tau sqrth $taucoarse $Param #launch offline+online parts

	
	FileList=$(ls NIRB_approximation*) 

	nev=$(echo $FileList|cut -d ' ' -f 1|cut -d '_' -f 4|cut -d '.' -f 1)
	echo nev: $nev

	nbFile=$(ls NIRB_approximation*|wc -l) #convert from vtu to vtk to be readable in FreeFem++ ( meshio 4.0.4 )
	
	for (( c=0; c<$nbFile; c++ ))
	do   
	    echo meshio convert, file number: $c
	    meshio-convert NIRB_approximation_${c}_${nev}.vtu NIRB_approximation_${c}_${nev}.vtk
	    meshio-ascii NIRB_approximation_${c}_${nev}.vtk
	done

	## error computation in FreeFem++
	
	nnref=$(echo $finemesh |cut -d ' ' -f $values)
	echo sizemesh $nnref
	FreeFem++-nw Crank_EulerPhiV2Adj.edp -tau $tau -nnref $nnref -Param $Param -nev $nev
	rm NIRB_app*
	echo !!!!!!!!!! sqrt ... time $tau !!!!!!!!!!!!!
	
    done
done
