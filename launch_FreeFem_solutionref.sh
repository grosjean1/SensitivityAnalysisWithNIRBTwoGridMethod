#! /bin/bash
# Elise Grosjean
# Initialization

#fine setting
finetime="0.0025"
finemesh="560"

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
	mkdir -p ~/Codes/CodesPourArticleSensibiliteBisAvecu0EllipticV2/FineSnapshotsRef/$tau/$param/
	mkdir -p ~/Codes/CodesPourArticleSensibiliteBisAvecu0EllipticV2/FineSnapshotsPhiRef/$tau/$param/
	
	FreeFem++-nw Crank_EulerSensitivityinit.edp -tau $tau -nnref $nnref -Param $param -theta $theta

	mv Snapshoth* ~/Codes/CodesPourArticleSensibiliteBisAvecu0EllipticV2/FineSnapshotsRef/$tau/$param/
	mv SnapshotPhih* ~/Codes/CodesPourArticleSensibiliteBisAvecu0EllipticV2/FineSnapshotsPhiRef/$tau/$param/
	
    done
done
