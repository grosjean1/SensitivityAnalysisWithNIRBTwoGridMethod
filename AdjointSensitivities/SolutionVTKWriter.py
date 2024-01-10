# -*- coding: utf-8 -*-
## VTk writer with vtkXMLUnstructuredGridWriter

## Elise Grosjean
## 01/2021


import vtk
from vtk.util import numpy_support
from BasicTools.Containers import Filters


def numpyToVTKWrite(VTKBase,Solution, solutionName="NIRBapproximation.vtu",dataName="Velocity"):

        numpy_array = Solution
        
        p = VTKBase.GetPointData()
        VTK_data = numpy_support.numpy_to_vtk(num_array=numpy_array, deep=True, array_type=vtk.VTK_FLOAT)
        
        VTK_data.SetName(dataName)
        p.AddArray(VTK_data)
        
        out_fname = solutionName

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(out_fname)
        writer.SetInputData(VTKBase)
        writer.SetDataModeToAscii()
        writer.Write()
        print('\nfile ', out_fname, ' written\n' )
