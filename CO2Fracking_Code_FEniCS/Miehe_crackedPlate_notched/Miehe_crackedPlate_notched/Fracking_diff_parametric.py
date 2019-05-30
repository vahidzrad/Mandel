# Copyright (C) 2017 Corrado Maurini, Tianyi Li
#
# This file is part of FEniCS Gradient Damage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
from dolfin import *
from mshr import *
import sys, os, sympy, shutil, math
import numpy as np
import matplotlib.pyplot as plt

from Fracture_QS_TAO import Fracking





hsize =  0.005

colors_i = ['r', 'b', 'g','m','c','k']


E = 210.e3 # Young modulus
nu = 0.3 # Poisson ratio

ModelB= True 
law='AT1'



load_steps_list= [100]

for (k, load_steps) in enumerate(load_steps_list):
		ell_list = [2*hsize, 4*hsize]

		for (i, ell) in enumerate(ell_list):
		    	# Varying the hsize mesh size
		       	Fracking(E, nu, hsize, ell, law, ModelB, load_steps)










