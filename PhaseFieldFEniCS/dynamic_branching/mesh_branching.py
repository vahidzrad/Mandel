# FEnics code  Gmsh
# Mostafa Mollaali
# Vahid Ziaei-Rad


from fenics import *
from dolfin import *
from mshr import *
import sympy, sys, math, os, subprocess, shutil
from subprocess import call
from dolfin_utils.meshconvert import meshconvert


#=======================================================================================
# Input date
#=======================================================================================
hsize=0.125e-3

#=======================================================================================
# Geometry and mesh generation
#=======================================================================================
meshname="mesh"

# Generate a XDMF/HDF5 based mesh from a Gmsh string
geofile = \
        """
            lc = DefineNumber[ %g, Name "Parameters/lc" ];
            H = 40.0e-3;
            L = 100.0e-3;

            a = 50.0e-3;

            Point(1) = {0, 0, 0, 10*lc};
            Point(2) = {L, 0, 0, 10*lc};
            Point(3) = {L, H, 0, 10*lc};
            Point(4) = {0, H, 0, 10*lc};

            Point(5) = {0, H/2.0, 0, 1*lc};
            Point(6) = {L, H/2.0, 0, 1*lc};
            Point(7) = {L/2.0, H/2.0, 0, 1*lc};
            
            Line(1) = {1, 2};
            Line(2) = {2, 6};
            Line(3) = {6, 3};
            Line(4) = {3, 4};
            Line(5) = {4, 5};
            Line(6) = {5, 1};
            Line(7) = {5, 7};
            Line(8) = {7, 6};
            
            Line Loop(100) = {1, 2, -8, -7, 6};
            Line Loop(200) = {7, 8, 3, 4, 5};
            

            Plane Surface(1000) = {100};
            Plane Surface(2000) = {200};
            
            // Line(6) = {5, 6};
            
            // Line{6} In Surface{30};
            
            Physical Surface(1) = {1000};
            Physical Surface(2) = {2000};
            
            // Physical Line(101) = {6};

"""%(hsize)


subdir = "meshes/"
_mesh  = Mesh() #creat empty mesh object


if not os.path.isfile(subdir + meshname + ".xdmf"):
        if MPI.comm_world.rank == 0:
            # Create temporary .geo file defining the mesh
            if os.path.isdir(subdir) == False:
                os.mkdir(subdir)
            fgeo = open(subdir + meshname + ".geo", "w")
            fgeo.writelines(geofile)
            fgeo.close()
            # Calling gmsh and dolfin-convert to generate the .xml mesh (as well as a MeshFunction file)
            try:
                    subprocess.call(["gmsh", "-2", "-o", subdir + meshname + ".msh", subdir + meshname + ".geo"])
            except OSError:
                    print("-----------------------------------------------------------------------------")
                    print(" Error: unable to generate the mesh using gmsh")
                    print(" Make sure that you have gmsh installed and have added it to your system PATH")
                    print("-----------------------------------------------------------------------------")


            meshconvert.convert2xml(subdir + meshname + ".msh", subdir + meshname + ".xml", "gmsh")

            # Convert to XDMF
            MPI.barrier(MPI.comm_world)
            mesh = Mesh(subdir + meshname + ".xml")
            XDMF = XDMFFile(MPI.comm_world, subdir + meshname + ".xdmf")
            XDMF.write(mesh)
            XDMF.read(_mesh)
        
        if os.path.isfile(subdir + meshname + "_physical_region.xml") and os.path.isfile(subdir + meshname + "_facet_region.xml"):
            if MPI.comm_world.rank == 0:
                    mesh = Mesh(subdir + meshname + ".xml")
                    subdomains = MeshFunction("size_t", mesh, subdir + meshname + "_physical_region.xml")
                    boundaries = MeshFunction("size_t", mesh, subdir + meshname + "_facet_region.xml")
                    HDF5 = HDF5File(MPI.comm_world, subdir + meshname + "_physical_facet.h5", "w")
                    HDF5.write(mesh, "/mesh")
                    HDF5.write(subdomains, "/subdomains")
                    HDF5.write(boundaries, "/boundaries")
                    print("Finish writting physical_facet to HDF5")

        print("Mesh completed")

    # Read the mesh if existing
else:
        XDMF = XDMFFile(MPI.comm_world, subdir + meshname + ".xdmf")
        
        XDMF.read(_mesh)
