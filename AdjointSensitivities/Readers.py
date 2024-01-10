from pathlib import Path
import os

def Readmesh(meshFileName):


    assert isinstance(meshFileName, str)
    suffix = str(Path(meshFileName).suffix)
    if suffix == ".vtu":
        from BasicTools.IO.VtuReader import VtkToMesh as Read
        from BasicTools.IO.VtuReader import LoadVtuWithVTK 
    else: 
        raise ("FileName error!")

    mesh = Read(LoadVtuWithVTK(meshFileName))
    return mesh


def VTKReadmesh(meshFileName):
    """
    Read a vtu mesh to save the solution in VTK format
    and return mesh
    """
    assert isinstance(meshFileName, str)
    suffix = str(Path(meshFileName).suffix)
    if suffix == ".vtu": 
        from BasicTools.IO.VtuReader import LoadVtuWithVTK 
    else: 
        raise ("FileName error!")
    mesh = LoadVtuWithVTK(meshFileName)
    return mesh

def VTKReadToNp(SolutionName, tmpbaseFile, i):
    """
    Read a vtu solution and return np.array
    """
    from BasicTools.IO.VtuReader import LoadVtuWithVTK
    from vtk.numpy_interface import dataset_adapter as dsa
        
    data = LoadVtuWithVTK(tmpbaseFile + str(i) + ".vtu")
        
    npArray = dsa.WrapDataObject(data).GetPointData().GetArray(SolutionName)
    return npArray
