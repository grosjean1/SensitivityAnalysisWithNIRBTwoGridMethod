#! /bin/bash
# Elise Grosjean
# Initialization

#fine setting
finetime="0.005"
finemesh="280"

####################
## fine snapshots ##
####################

theta=1. #euler

for values in 1
do
    for param in 1
    do
	
	tau=$(echo $finetime | cut -d ' ' -f $values)
	echo time $tau
	nnref=$(echo $finemesh |cut -d ' ' -f $values)
	echo sizemesh $nnref
	mkdir -p ./FineSnapshotsRef/$tau/$param/
	mkdir -p ./FineAdjointRef/$tau/$param/

	FreeFem++-nw Crank_Eulerinit.edp -tau 0.0025 -nnref 560 -Param $param -theta $theta
	mv Snapshoth* ./FineSnapshotsRef/0.0025/$param/

	
	FreeFem++-nw AdjointCrank_EulerSensitivityinit.edp -tau $tau -nnref $nnref -Param $param -theta $theta
	mv SnapshotPhih* ./FineAdjointRef/$tau/$param/
	
    done
done
