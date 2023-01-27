# Sensitivity Analysis With NIRB Two-Grid Method
Sensitivity Analysis Heat equation With NIRB Two-Grid Method as in https://arxiv.org/abs/2301.00761

## NIRB with heat equation

launch_FreeFem.sh:
Create Folders with the snapshots

launch_FreeFem_solutionref.sh:
Refence solution

launch_NIRB: #user can change the value of parameter line 20 (" for Param in 1 ")
Offline+Online NIRB two-grid algorithm

with Rectification:
     NirbGreedyOfflineRectif.py
     NirbGreedyOnlineRectif.py

with Gaussian process regression:
     NirbGreedyOfflineRectifGP.py
     NirbGreedyOnlineRectifGP.py

Careful:
	offline (Greedy algorithm) and online part Mesh reader with Basictools Warning: basic-tools/src must be added to the pythonpath (https://gitlab.com/drti/basic-tools/-/tree/public_master/src/BasicTools)

	Data conversion for FreeFem++ with meshio 4.0.4 https://pypi.org/project/meshio/