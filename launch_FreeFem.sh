#! /bin/bash
# Elise Grosjean
# Initialization

#fine setting
finetime="0.01 0.02 0.05 0.1"
finemesh="140 70 30 15"

#coarse setting
coarsetime="0.1 0.1414 0.22 0.32" #sqrth
coarsemesh="15 10 7 5"

####################
## coarse snapshots ##
####################

theta=0.5 #Crank-Nicolson
for values in 2 3 4
do
    for param in 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5
    do

	tau=$(echo $coarsetime | cut -d ' ' -f $values) #time step 
	echo time $tau
	nnref=$(echo $coarsemesh |cut -d ' ' -f $values) #size
	echo sizemesh $nnref
	mkdir -p ~/Codes/CodesPourArticleSensibiliteBisAvecu0EllipticV2/CoarseSnapshots/hdiv2/$tau/$param/
	mkdir -p ~/Codes/CodesPourArticleSensibiliteBisAvecu0EllipticV2/CoarseSnapshotsPhi/hdiv2/$tau/$param/
	FreeFem++-nw Crank_EulerSensitivityinit.edp -tau $tau -nnref $nnref -Param $param -theta $theta
	mv Snapshoth* ~/Codes/CodesPourArticleSensibiliteBisAvecu0EllipticV2/CoarseSnapshots/hdiv2/$tau/$param/
	mv SnapshotPhih* ~/Codes/CodesPourArticleSensibiliteBisAvecu0EllipticV2/CoarseSnapshotsPhi/hdiv2/$tau/$param/
    done
done


####################
## fine snapshots ##
####################

theta=1. #euler

for values in 1 2 3 4
do
    for param in 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5
    do	 
	tau=$(echo $finetime | cut -d ' ' -f $values)
	echo time $tau
	nnref=$(echo $finemesh |cut -d ' ' -f $values)
	echo sizemesh $nnref
	mkdir -p ~/Codes/CodesPourArticleSensibiliteBisAvecu0EllipticV2/FineSnapshots/$tau/$param/
	mkdir -p ~/Codes/CodesPourArticleSensibiliteBisAvecu0EllipticV2/FineSnapshotsPhi/$tau/$param/
	FreeFem++-nw Crank_EulerSensitivityinit.edp Crank_Eulerinit.edp -tau $tau -nnref $nnref -Param $param -theta $theta
	mv Snapshoth* ~/Codes/CodesPourArticleSensibiliteBisAvecu0EllipticV2/FineSnapshots/$tau/$param/
	mv SnapshotPhih* ~/Codes/CodesPourArticleSensibiliteBisAvecu0EllipticV2/FineSnapshotsPhi/$tau/$param/
    done
done

