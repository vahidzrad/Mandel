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

hsize = 0.8
L = 50.0
H = 9.8

subdir = "meshes/"
meshname="fracking_hsize%g" % (hsize)
mesh = Mesh(subdir + meshname + ".xml")

#mesh = Mesh('mesh.xml')
#mesh = Mesh('meshes/fracking_hsize'+str(float(hsize))+'.xml')
mesh_fun = MeshFunction("size_t", mesh, subdir + meshname + "_facet_region.xml")

# Define Space
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)
WW = FunctionSpace(mesh, 'DG', 1)

p, q = TrialFunction(V), TestFunction(V)
dT, T_ = TrialFunction(V), TestFunction(V)
u, v = TrialFunction(W), TestFunction(W)

# Introduce manually the material parameters
Gc =  42.47e3  # MPa.mm      Hint: J/m^2=1e-3 MPa.mm
l = 4*hsize
#lmbda = 121.1538e3
#mu = 80.7692e3

T0 = Constant(680.)  
E = 340e3 #MPa
nu = 0.22
lmbda  = Constant(E*nu/((1+nu)*(1-2*nu)))
mu = Constant(E/2/(1+nu)) 
rho = 2700.e-9     # density #kg/mm^3
alpha = Constant(8.0e-6) # thermal expansion coefficient #K^-1
kappa  = Constant(alpha*(2*mu + 3*lmbda)) 
cV = Constant(961.5e3)*rho # specific heat per unit volume at constant strain #J/(kgK)= 1e3 MPa*mm^3/(kgK)
k = Constant(6.)  # thermal conductivity #W/(mK)=J/(mKs)= MPa*mm^2/(Ks).
deltaT  = hsize**2 * rho*cV/k
print('DeltaT', deltaT)

# Constituive functions
def epsilon(u):
    return sym(grad(u))
#def sigma(u):
#    return 2.0*mu*epsilon(u)+lmbda*tr(epsilon(u))*Identity(len(u))
def sigma(u, dT):
    return (lmbda*tr(epsilon(u)) - kappa*dT)*Identity(2) + 2*mu*epsilon(u)

def psi(u):
    return 0.5*(lmbda+mu)*(0.5*(tr(epsilon(u))+abs(tr(epsilon(u)))))**2+\
           mu*inner(dev(epsilon(u)),dev(epsilon(u)))		
def H(uold,unew,Hold):
    return conditional(lt(psi(uold),psi(unew)),psi(unew),Hold)
		
# Boundary conditions
top = CompiledSubDomain("near(x[1], 4.9) && on_boundary")
bot = CompiledSubDomain("near(x[1], -4.9) && on_boundary")
left = CompiledSubDomain("near(x[0], -25.) && on_boundary")
right = CompiledSubDomain("near(x[0], 25.) && on_boundary")


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


def Crack(x):
    return abs(x[1]) < 1e-03 and x[0] <= 0.0

load = Expression("t", t = 0.0, degree=1)

bc_u_bot= DirichletBC(W, Constant((0.0,0.0)), bot)
bc_u_top = DirichletBC(W.sub(1), load, top)
bc_u_pt_l = DirichletBC(W, Constant([0.,0.]), pinpoint_l, method='pointwise')
bc_u_pt_r = DirichletBC(W, Constant([0.,0.]), pinpoint_r, method='pointwise')
bc_u = [bc_u_pt_l, bc_u_pt_r]

bc_phi = [DirichletBC(V, Constant(0.0), right)]

bc_T_top = DirichletBC(V, Constant(300.0), top)
bc_T_bot = DirichletBC(V, Constant(300.0), bot)
bc_T_left = DirichletBC(V, Constant(300.0), left)
bc_T_right = DirichletBC(V, Constant(300.0), right)

bc_T = [bc_T_top, bc_T_bot, bc_T_left, bc_T_right]

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top.mark(boundaries,1)
ds = Measure("ds")(subdomain_data=boundaries)
n = FacetNormal(mesh)

# Variational form
unew, uold = Function(W), Function(W)
pnew, pold, Hold = Function(V), Function(V), Function(V)	
Tnew, Told = Function(V), Function(V)
Told = project(T0, V)

E_du = ((1.0-pold)**2)*inner(grad(v),sigma(u, Told))*dx
E_phi = (Gc*l*inner(grad(p),grad(q))+((Gc/l)+2.0*H(uold,unew,Hold))\
            *inner(p,q)-2.0*H(uold,unew,Hold)*q)*dx
		
E_T = (cV*(dT-Told)/deltaT*T_ +  dot(k*grad(dT), grad(T_)))*dx
	
p_disp = LinearVariationalProblem(lhs(E_du), rhs(E_du), unew, bc_u)
p_phi = LinearVariationalProblem(lhs(E_phi), rhs(E_phi), pnew, bc_phi)
solver_disp = LinearVariationalSolver(p_disp)
solver_phi = LinearVariationalSolver(p_phi)
		
p_T = LinearVariationalProblem(lhs(E_T), rhs(E_T), Tnew, bc_T)
solver_T = LinearVariationalSolver(p_T)

# Initialization of the iterative procedure and output requests
t = 0
u_r = 0.007
u_T = 1.
#deltaT  = 0.1
tol = 1e-3
conc_f = File ("./ResultsDir/phi.pvd")
conc_T = File ("./ResultsDir/Temp.pvd")

fname = open('ForcevsDisp.txt', 'w')

# Staggered scheme
while t<=1.0:
    t += deltaT
    #if t >=0.7:
    #    deltaT = deltaT #0.0001 #Edited by Mostafa
    load.t=t*u_T
    iter = 0
    err = 1

    while err > tol:
        iter += 1
        solver_disp.solve()
        solver_phi.solve()
        solver_T.solve() 

	

        err_u = errornorm(unew,uold,norm_type = 'l2',mesh = None)
        err_phi = errornorm(pnew,pold,norm_type = 'l2',mesh = None)
        err_T = errornorm(Tnew,Told,norm_type = 'l2',mesh = None)
	
        err = max(err_u,err_phi, err_T)
        print('err_u', err_u)
        print('err_phi', err_phi)
        print('err_T', err_T)

        uold.assign(unew)
        pold.assign(pnew)
        Told.assign(Tnew)
        Hold.assign(project(psi(unew), WW))

        if err < tol:
		
            print ('Iterations:', iter, ', Total time', t)

            if round(t*1e4) % 10 == 0:
                conc_f << pnew 
                conc_T << Tnew


                Traction = dot(sigma(unew, Told),n)
                fy = Traction[1]*ds(1)
                fname.write(str(t*u_r) + "\t")
                fname.write(str(assemble(fy)) + "\n")
	    	    
fname.close()
print ('Simulation completed') 
