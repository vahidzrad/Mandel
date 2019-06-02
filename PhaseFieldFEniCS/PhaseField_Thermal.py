# Phase field fracture implementation in FEniCS    
# The code is distributed under a BSD license     
      
# If using this code for research or industrial purposes, please cite:
# Hirshikesh, S. Natarajan, R. K. Annabattula, E. Martinez-Paneda.
# Phase field modelling of crack propagation in functionally graded materials.
# Composites Part B: Engineering 169, pp. 239-248 (2019)
# doi: 10.1016/j.compositesb.2019.04.003
      
# Emilio Martinez-Paneda (mail@empaneda.com)
# University of Cambridge

# Preliminaries and mesh
from dolfin import *
import numpy as np

L = 50.0	# Width: mm (Chu 2017-3.3)
H = 9.8		# Height: mm (Chu 2017-3.3)

subdir = "meshes/"
meshname = "mesh" # "fracking_hsize%g" % (hsize)

#mesh = Mesh('mesh.xml')
mesh = Mesh(subdir + meshname + ".xml")
mesh_fun = MeshFunction("size_t", mesh, subdir + meshname + "_facet_region.xml")

# Define Space
V_u = VectorFunctionSpace(mesh, 'CG', 1)
V_d = FunctionSpace(mesh, 'CG', 1)
#WW = FunctionSpace(mesh, 'DG', 0)

u_, u, u_t = Funciton(V_u), TrialFunction(V_u), TestFunction(V_u)
d_, d, d_t = Function(V_d), TrialFunction(V_d), TestFunction(V_d)
T_, T, T_t = Function(V_d), TrialFunction(V_d), TestFunction(V_d)

# Introduce manually the material parameters
E = 380.0e6		# Young's modulus: MPa (Chu 2017-4.1)
nu = 0.25		# Poisson's ratio: - (Chu 2017-4.1)
Gc = 26.95		# critical energy release rate: MPa-mm (Chu 2017-4.1)

l = 0.4			# length scale: mm (Chu 2017-4.1)
hsize = l/2.		# mesh size: mm (Chu 2017-4.1)

Ts = Constant(680.)  	# initial temperature of slab: K (Chu 2017-3.3)
Tw = Constant(300.)	# temperature of surface contacted with water: K (Chu 2017-3.3)

lmbda  = Constant(E*nu/((1+nu)*(1-2*nu)))		# Lamé constant: MPa (conversion formulae)
mu = Constant(E/(2*(1+nu))) 				# shear modulus: MPa (conversion formulae)

rho = 3.9e-6						# density: kg/m^3 (Chu 2017-4.1)

alpha = Constant(6.6e-6)				# linear expansion coefficient: 1/K (Chu 2017-4.1)
# kappa  = Constant(alpha*(2*mu + 3*lmbda))		# ?

c = Constant(961.5e6)					# specific heat of material: #J/(kgK) (Chu 2017-4.1)
k = Constant(21.0e3)					# thermal conductivity: #W/(mK)=J/(mKs) (Chu 2017-4.1)
deltaT = rho * c * height**2 / k			# source: (Chu2017-3.3)
print('DeltaT', deltaT)

# Constituive functions
# strain
def epsilon(u_):
	return sym(grad(u_))

def epsilonT(u_, T_):
	return alpha * (T_ - Ts) * Identity(len(u_))

def epsilone(u_, T_):
	return epsilon(u_) - epsilonT(u_, T_)

# stress
def sigma(u_): # not applicable
	return lmbda * tr(epsilon(u_)) * Identity(len(u_)) + 2.0 * mu * epsilon(u_)

def sigma(u_, T_): # no decomposition
	return lmbda * tr(epsilone(u_, T_)) * Identity(len(u_)) + 2.0 * mu * (epsilone(u_, T_))

def sigmap(u_, T_): # sigma_+ for Amor's model
	return (lmbda + 2.0 * mu / 3.0) * (tr(epsilone(u_, T_)) + abs(tr(epsilone(u_, T_))))/2.0 * Identity(len(u_)) \
		+ 2.0 * mu * dev(epsilone(u_, T_))
def sigman(u_, T_): # sigma_- for Amor's model
	return (lmbda + 2.0 * mu / 3.0) * (tr(epsilone(u_, T_)) - abs(tr(epsilone(u_, T_))))/2.0 * Identity(len(u_))

# strain energy
def psip(u_, T_):
	return (lmbda/2.0 + mu/3.0) * ((tr(epsilone(u_, T_)) + abs(tr(epsilone(u_, T_))))/2.0)**2\
	+ mu * inner(dev(epsilone(u_, T_)), dev(epsilone(u_, T_)))

# Boundary conditions
top = CompiledSubDomain("near(x[1], H/2.) && on_boundary")
bot = CompiledSubDomain("near(x[1], -H/2.) && on_boundary")
left = CompiledSubDomain("near(x[0], -L) && on_boundary")
right = CompiledSubDomain("near(x[0], L) && on_boundary")

class Pinpoint(SubDomain):
    TOL = 1e-3
    def __init__(self, coords):
        self.coords = np.array(coords)
        SubDomain.__init__(self)
    def move(self, coords):
        self.coords[:] = np.array(coords)
    def inside(self, x, on_boundary):
        TOL = 1e-3
        return np.linalg.norm(x-self.coords) < TOL

pinpoint_l = Pinpoint([0.,0.])
pinpoint_r = Pinpoint([L,0.])

load = Expression("t", t = 0.0, degree=1)

# Boundary conditions for u
bc_u_bot= DirichletBC(V_u, Constant((0.0,0.0)), bot)
bc_u_top = DirichletBC(V_u.sub(1), load, top)
bc_u_pt_left = DirichletBC(V_u, Constant([0.,0.]), pinpoint_l, method='pointwise')
bc_u_pt_right = DirichletBC(V_u, Constant([0.,0.]), pinpoint_r, method='pointwise')
bc_u = [bc_u_pt_left, bc_u_pt_right]

bc_d = [DirichletBC(V_d, Constant(0.0), right)]

# Boundary conditions for T
bc_T_top = DirichletBC(V_d, Tw, top) # Vahid: Tw or Ts?
bc_T_bot = DirichletBC(V_d, Tw, bot)
bc_T_left = DirichletBC(V_d, Tw, left)
bc_T_right = DirichletBC(V_d, Tw, right)
bc_T = [bc_T_top, bc_T_bot, bc_T_left, bc_T_right]

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top.mark(boundaries,1)
ds = Measure("ds")(subdomain_data=boundaries)
n = FacetNormal(mesh)

# Variational form
E_u = (1.0-d_)**2.0 * inner(sigmap(u, T), epsilone(u_t, T_t)) * dx + inner(sigman(u, T), epsilone(u_t, T_t)) * dx
E_d = (3.0/8.0) * Gc * (d_t/l * dx + 2.0 * l * inner(grad(d), grad(d_t)) * dx) - 2.0 * (1.0 - d) * psip(u_, T_) * d_t * dx

# T0 = project(Ts, V_d)
T0 = interpolate(Expression('T_int', T_int = Ts, degree=1), V_d)
E_T = (1.0 - d_)**2 * rho * c * (T - T0) / deltaT * T_t * dx - (1.0 - d_)**2 * k * inner(grad(T), grad(T_t)) * dx

problem_u = LinearVariationalProblem(lhs(E_u), rhs(E_u), u_, bc_u)
problem_d = LinearVariationalProblem(lhs(E_d), rhs(E_d), d_, bc_d)
solver_u = LinearVariationalSolver(problem_u)
solver_d = LinearVariationalSolver(problem_d)

problem_T = LinearVariationalProblem(lhs(E_T), rhs(E_T), T_, bc_T)
solver_T = LinearVariationalSolver(problem_T)

# Initialization of the iterative procedure and output requests
t = 0
u_r = 0.007
u_T = 1.
#deltaT  = 0.1
tol = 1e-3
conc_d = File ("./ResultsDir/d.pvd")
conc_T = File ("./ResultsDir/T.pvd")

fname = open('ForcevsDisp.txt', 'w')

# Staggered scheme
while t<=1.0:
    t += deltaT
    #if t >=0.7:
    #    deltaT = deltaT #0.0001 #Edited by Mostafa
    load.t=t*u_T
    iter = 0
    err = 1.0

    while err > tol:
        iter += 1
        solver_u.solve()
        solver_d.solve()
        solver_T.solve() 

        err_u = errornorm(unew,uold,norm_type = 'l2',mesh = None)
        err_d = errornorm(pnew,pold,norm_type = 'l2',mesh = None)
        err_T = errornorm(Tnew,Told,norm_type = 'l2',mesh = None)
	
        err = max(err_u, err_d, err_T)
        print('err_u: ', err_u)
        print('err_d: ', err_d)
        print('err_T: ', err_T)

        uold.assign(u_)
        pold.assign(pnew)
        Told.assign(Tnew)
        Hold.assign(project(psi(unew), WW))

        if err < tol:
		print ('Iterations:', iter, ', Total time', t)
		if round(t*1e4) % 10 == 0:
                	conc_d << d_ 
	                conc_T << T_
	
        	        Traction = dot(sigma(u_, T_), n)
                	fy = Traction[1]*ds(1)
				
	                fname.write(str(t*u_r) + "\t")
        	        fname.write(str(assemble(fy)) + "\n")

fname.close()
print ('Simulation completed')
