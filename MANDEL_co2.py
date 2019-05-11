# FEnics code  Variational Fracture Mechanics
#
# A static solution of the variational fracture mechanics problems using the regularization AT2/AT1
# authors:
# corrado.maurini@upmc.fr
# Mostafa Mollaali
# Vahid


from fenics import *
from dolfin import *
from mshr import *
from dolfin_utils.meshconvert import meshconvert
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy, sys, math, os, subprocess, shutil
import petsc4py
petsc4py.init()
from petsc4py import PETSc

from numpy import loadtxt 
from EOS import EqOfState
from EOS_N import EqOfState_N

#=======================================================================================
	# Input date
#=======================================================================================
# Geometry
L = 2.5 # m,length
H = 1. # m, height
Lx= 50
Ly= 20
meshname="fracking_hsize%g" % (Lx)

# Material constants
PlaneStress= False
E = 1.e6 # Young modulus Pa
nu = 0.2 # Poisson ratio
G=E/(2.*(1.+nu))
#fixed stress
k_fixed2D= E/(2.*(1.+nu)*(1.-2.*nu))


# Loading
body_force = Constant((0.,0.))  # bulk load
load_min = 0. # load multiplier min value
load_max = 1. # load multiplier max value
load_steps = 600# number of time steps
DeltaT = 1.e-2 
#====================================================================================
# To define  pressure field
#====================================================================================
# Input data: Table 3.2 of Chukwudozie 2016, page 104
biot=1.0          # Biot coefficient
kappa= 1.0        #is the permeability of the rock #m^2
mu_dynamic= 1.e6    #is the dynamic viscosity of the fluid #MPa.s

nu_u = 0.5       #undrained Poisson ratio	
B=1.0            #the Skempton coefficient
F=2.5e6          #Pa.m=N/m, Load, The application of a load (2F) causes an instantaneous and uniform pressure increase throughout the domain

phi=0.01
ka=7.
kb=1.


c=2.*kappa*B**2.*G*(1.-nu)*(1.+nu_u)**2/(9.*mu_dynamic*(1.-nu_u)*(nu_u-nu))
#print c/L**2*1e-6
Uy_analytical = loadtxt("verticalDisp.txt", comments="#", delimiter=",", unpack=False)
#print Uy_analytical
#=======================================================================================
# Geometry and mesh generation
#=======================================================================================
mesh = RectangleMesh(Point(0., 0.), Point(L, H), Lx, Ly)
ndim = mesh.geometry().dim() # get number of space dimensions
zero_v = Constant((0.,)*ndim)
#=======================================================================================
# strain, stress and strain energy for Isotropic and Amor's model
#=======================================================================================
def eps(u_):
	return sym(grad(u_))
#----------------------------------------------------------------------------------------
def sigma0(u_):
	Id = Identity(len(u_))
	return 2.0*mu*eps(u_) + lmbda*tr(eps(u_))*Id
	
#----------------------------------------------------------------------------------------
def psi_0(u_):
	return  0.5 * lmbda * tr(eps(u_))**2 + mu * eps(u_)**2

#=======================================================================================
# others definitions
#=======================================================================================
prefix = "L%s-H%.2f-steps%s,mu%s"%(L,H, load_steps,mu_dynamic)
save_dir = "Fracking_result/" + prefix + "/"
if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)


# plane strain or plane stress
if not PlaneStress:  # plane strain
	lmbda = E*nu/((1.0+nu)*(1.0-2.0*nu))
else:  # plane stress
	lmbda = E*nu/(1.0-nu**2)
# shear modulus
mu = E / (2.0 * (1.0 + nu)) 
#=======================================================================================
# Define boundary sets for boundary conditions
#=======================================================================================
class Right(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and abs(x[0] - L) < DOLFIN_EPS
class Left(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and abs(x[0]) < DOLFIN_EPS
class Top(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and abs(x[1] - H) < DOLFIN_EPS
class Bottom(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and abs(x[1]) < DOLFIN_EPS

# Neumann boundary condition on top edge
class MyExpr(Expression):
	    def eval(self, values, x):
		if abs(x[1] - H)<DOLFIN_EPS:
			values[0] = Uy_analytical[i_t]#-H/(4.*G*L)*(2.*(1.-nu)*F+biot*(1.-2.*nu)*assemble(P_*ds(3)))#
my_expr = MyExpr(degree=1)


# Initialize sub-domain instances
right=Right()
left=Left()
top=Top()
bottom=Bottom()

# define meshfunction to identify boundaries by numbers
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(9999)
right.mark(boundaries, 1) # mark top as 1
left.mark(boundaries, 2) # mark top as 2
top.mark(boundaries, 3) # mark top as 3
bottom.mark(boundaries, 4) # mark bottom as 4


# Define new measure including boundary naming 
ds = Measure("ds")[boundaries] # left: ds(1), right: ds(2)
#=======================================================================================
# Variational formulation
#=======================================================================================
# Create function space for 2D elasticity + Damage
V_u = VectorFunctionSpace(mesh, "CG", 1)
V_p = FunctionSpace(mesh, "CG", 1)

# Define the function, test and trial fields
u_, u, u_t = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
P_, P, P_t=  Function(V_p), TrialFunction(V_p), TestFunction(V_p),
rho = Function(V_p)  #density kg/m^3
N = Function(V_p)    #N=R*T*(delta**2 * phi_r_delta_delta +2*delta*phi_r_delta + 1) # J/(Kg*K)*K-->m^2/s^2

P_k = interpolate(Constant(0.), V_p) #intial pressure  (eq41, Andro Mikelic  Bin Wang  Mary F Wheeler2014)
u_k = interpolate(zero_v, V_u) # intial disp (eq42-43, Andro Mikelic  Bin Wang  Mary F Wheeler2014)

P_0 = interpolate(Expression("F*B*(1.+nu_u)/(3.*L)", F=F,nu_u=nu_u, B=B, L=L, degree=1), V_p) #intial pressure  (eq41, Andro Mikelic  Bin Wang  Mary F Wheeler2014)
u_0 = interpolate(Expression(('F*nu_u/(2.*G)', '-F*(1-nu_u)*H/(2.*G*L)'),F=F,nu_u=nu_u, G=G, L=L, H=H, degree=2), V_u) # intial disp (eq42-43, Andro Mikelic  Bin Wang  Mary F Wheeler2014)

Init_Pressure=0.5e6#F*B*(1.+nu_u)/(3.*L)
Rho_intial=EqOfState(Init_Pressure) #initial density according to initial pressure(Pa?)
print "Rho_intial=",Rho_intial
N_intial=EqOfState_N([Rho_intial]) #initial N(m^2/s^2) according to initial density(kg/m^3)



print "N_intial=",N_intial[0]

#------------------------------------------------------------------------------------------
N_intial=Constant(N_intial[0])
Rho_intial=Constant(Rho_intial)



rho = interpolate(Expression('Rho', Rho = Rho_intial, degree=1), V_p)
rho_0 = interpolate(Expression('Rho', Rho = Rho_intial, degree=1), V_p)
N = interpolate(Expression('N', N= N_intial, degree=1), V_p)
#=======================================================================================
# Dirichlet boundary condition for a traction test boundary
#=======================================================================================
## bc - u (imposed displacement)
Gamma_u_0= DirichletBC(V_u.sub(1), Constant(0.0), boundaries, 4) #the bottom side in y direction is fixed
Gamma_u_1= DirichletBC(V_u.sub(0), Constant(0.0), boundaries, 2) #the left side in y direction is fixed
Gamma_u_2= DirichletBC(V_u.sub(1), my_expr, boundaries, 3) #the top side in y direction is loaded
bc_u =[ Gamma_u_0, Gamma_u_1, Gamma_u_2]

## bc - P (imposed pressure)
Gamma_P_0 = DirichletBC(V_p, 0.0, boundaries, 1)
bc_P = [Gamma_P_0]
#====================================================================================
# Define  problem and solvers
#====================================================================================
#1./M=0
#Pressure = (biot**2/k_fixed2D)/DeltaT *(P_*P - P*P_0)*dx- (biot**2/k_fixed2D)/DeltaT *(P_k*P - P*P_0)*dx+ (kappa/mu_dynamic)*inner(nabla_grad(P_), nabla_grad(P))*dx+biot/DeltaT*(div(u_k)*P-div(u_0)*P)*dx #check Chukwudozie 2016, page 76 eq:3.16. M=inf, alpha=1, kappa=k_fixed2D (I suppose)
# added by Vahid
#Pressure = phi/DeltaT*(rho-rho_0)*P*dx+rho/DeltaT*(div(u_)-div(u_0))*P*dx+rho*(kappa/mu_dynamic)*inner(nabla_grad(P_), nabla_grad(P))*dx

#Pressure = phi/(N*DeltaT)*(P_-P_0)*P*dx+rho/DeltaT*(div(u_)-div(u_0))*P*dx+rho*(kappa/mu_dynamic)*inner(nabla_grad(P_), nabla_grad(P))*dx
Pressure = (phi/N+biot*rho/k_fixed2D)/DeltaT *(P_*P - P*P_0)*dx + (kappa/mu_dynamic*rho)*inner(nabla_grad(P_), nabla_grad(P))*dx+rho/DeltaT*(div(u_k)*P-div(u_0)*P)*dx - (biot*rho/k_fixed2D)/DeltaT *(P_k*P - P*P_0)*dx
#------------------------------------------------------------------------------------
elastic_energy = psi_0(u_)*dx-biot*P_*div(u_)*dx #eq (1.5)paper :PHASE-FIELD MODELING OF PRESSURIZED FRACTURES IN A POROELASTIC MEDIUM, ANDRO MIKELIC et al.
# added by Vahid
#elastic_energy = (psi_0(u_)-(biot-1)*P_*div(u_)+inner(nabla_grad(P_), u_))*dx #eq (1.5)paper :PHASE-FIELD MODELING OF PRESSURIZED FRACTURES IN A POROELASTIC MEDIUM, ANDRO MIKELIC et al.
total_energy = elastic_energy


# Residual and Jacobian of elasticity problem
Du_total_energ = derivative(total_energy, u_, u_t)
J_u = derivative(Du_total_energ, u_, u)

#Jacobian of pressure problem
J_p  = derivative(Pressure, P_, P_t) 
# Variational problem for the displacement
problem_u = NonlinearVariationalProblem(Du_total_energ, u_, bc_u, J_u)

# Parse (PETSc) parameters
parameters.parse()

# Set up the solvers                                        
solver_u = NonlinearVariationalSolver(problem_u)   
prm = solver_u.parameters
prm["newton_solver"]["absolute_tolerance"] = 1E-6
prm["newton_solver"]["relative_tolerance"] = 1E-6
prm["newton_solver"]["maximum_iterations"] = 200
prm["newton_solver"]["relaxation_parameter"] = 1.0
prm["newton_solver"]["preconditioner"] = "default"
prm["newton_solver"]["linear_solver"] = "mumps"              


problem_pressure = NonlinearVariationalProblem(Pressure, P_, bc_P, J=J_p)
solver_pressure = NonlinearVariationalSolver(problem_pressure)       
#=======================================================================================
# To store results
#=======================================================================================
load_multipliers = np.linspace(load_min,load_max,load_steps)
file_u = File(save_dir+"/u.pvd") # use .pvd if .xdmf is not working
file_p = File(save_dir+"/p.pvd") 
 

colors_i = ['b','r','y','c','m','k']
colorID=0
#=======================================================================================
# Solving at each timestep
#=======================================================================================
pressure_center= np.zeros((len(load_multipliers),2))
dispVertical_center= np.zeros((len(load_multipliers),2))
pressure = np.zeros((Lx+1,1))
pltUX = np.zeros((Lx+1,1))
xcoord = np.zeros((Lx+1,1))
tolerance=1e-5
max_iterations=100
plt.figure(1)
for (i_t, t) in enumerate(load_multipliers):
	print"\033[1;32m--- Time step %d: time = %g---\033[1;m" % (i_t, c/L**2*i_t*DeltaT)
	# solve elastic problem

	iter= 0 ; err_P = 1;  err_u = 1
	# Iterations
	while  err_P>tolerance:# and err_u>tolerance:# and iter<max_iterations:
		# solve pressure problem
		solver_pressure.solve()
		err_P = (P_.vector() - P_k.vector()).norm('l2')
		if mpi_comm_world().rank == 0:
			print "Iteration:  %2d, pressure_Error: %2.8g, P_max: %.8g MPa" %(iter, err_P, P_.vector().max()/1e6)

		rho=EqOfState(P_.vector().get_local().shape) 	#set the new density according new pressure
	        N=EqOfState_N(rho)			#set the new N according new pressure
		#------------------------------------------------------------------------------------------
		solver_u.solve()
		err_u = (u_k.vector() - u_0.vector()).norm('l2')
		if mpi_comm_world().rank == 0:
				print "Iteration:  %2d, displacement_Error: %2.8g, u_max: %.8g m" %(iter, err_u, u_.vector().max())
		#------------------------------------------------------------------------------------------
		u_k.vector()[:] = u_.vector()
		P_k.vector()[:] = P_.vector()
		my_expr = MyExpr(degree=1)
		iter += 1
		#------------------------------------------------------------------------------------------
	u_0.vector()[:] = u_.vector()
	P_0.vector()[:] = P_.vector()
	rho_0=EqOfState(P_.vector().get_local().shape) 	#set the new density according new pressure
	
        # Dump solution to file 
    	file_u << u_
	file_p << P_ 

