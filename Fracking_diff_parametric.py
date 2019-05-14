from dolfin import *
from mshr import *
import sys, os, sympy, shutil, math
import numpy as np
import matplotlib.pyplot as plt

from Fracking_tao_diffusion import Fracking



E = 1.e6 # Young modulus Pa
nu = 0.2 # Poisson ratio
mu_dynamic_list= [1.] #is the dynamic viscosity of the fluid #Pa.s

for (k, mu_dynamic) in enumerate(mu_dynamic_list):
                Fracking(E, nu, k, mu_dynamic)


#### Remove the .pyc file ####
MPI.barrier(MPI.comm_world)
if MPI.rank(MPI.comm_world) == 0:
    os.remove("Fracking_tao_diffusion.pyc")