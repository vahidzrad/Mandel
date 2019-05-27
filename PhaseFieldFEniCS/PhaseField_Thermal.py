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
hsize=0.01

#mesh = Mesh('mesh.xml')
mesh = Mesh('meshes/fracking_hsize'+str(float(hsize))+'.xml')
mesh_fun = MeshFunction("size_t", mesh,"meshes/fracking_hsize"+str(float(hsize))+"_facet_region.xml")

# Define Space
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)
WW = FunctionSpace(mesh, 'DG', 0)
p, q = TrialFunction(V), TestFunction(V)
#Added by Mostafa--------------------------
dT, T_ = TrialFunction(V), TestFunction(V)
#------------------------------------------
u, v = TrialFunction(W), TestFunction(W)

# Introduce manually the material parameters
Gc =  2.7
l = 0.015
lmbda = 121.1538e3
mu = 80.7692e3

#Added by Mostafa--------------------------
T0 = Constant(293.)  
E = 70e3
nu = 0.3
#lmbda  = Constant(E*nu/((1+nu)*(1-2*nu)))
#mu = Constant(E/2/(1+nu)) 
rho = 2700.     # density
alpha = 2.31e-5 # thermal expansion coefficient
kappa  = Constant(alpha*(2*mu + 3*lmbda)) 
cV = Constant(910e-6)*rho # specific heat per unit volume at constant strain
k = Constant(237e-6)  # thermal conductivity
deltaT  = 0.1

#------------------------------------------

# Constituive functions
def epsilon(u):
    return sym(grad(u))
#def sigma(u):
#    return 2.0*mu*epsilon(u)+lmbda*tr(epsilon(u))*Identity(len(u))
#Added by Mostafa--------------------------
def sigma(u, dT):
    return (lmbda*tr(epsilon(u)) - kappa*dT)*Identity(2) + 2*mu*epsilon(u)	
#------------------------------------------
	
def psi(u):
    return 0.5*(lmbda+mu)*(0.5*(tr(epsilon(u))+abs(tr(epsilon(u)))))**2+\
           mu*inner(dev(epsilon(u)),dev(epsilon(u)))		
def H(uold,unew,Hold):
    return conditional(lt(psi(uold),psi(unew)),psi(unew),Hold)
		
# Boundary conditions
top = CompiledSubDomain("near(x[1], 0.5) && on_boundary")
bot = CompiledSubDomain("near(x[1], -0.5) && on_boundary")
def Crack(x):
    return abs(x[1]) < 1e-03 and x[0] <= 0.0
load = Expression("t", t = 0.0, degree=1)
bcbot= DirichletBC(W, Constant((0.0,0.0)), bot)
bctop = DirichletBC(W.sub(1), load, top)
bc_u = [bcbot, bctop]
bc_phi = [DirichletBC(V, Constant(1.0), Crack)]
#Added by Mostafa--------------------------		
bc_T = [DirichletBC(V, Constant(303.0), Crack)]
#------------------------------------------


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top.mark(boundaries,1)
ds = Measure("ds")(subdomain_data=boundaries)
n = FacetNormal(mesh)

# Variational form
unew, uold = Function(W), Function(W)
pnew, pold, Hold = Function(V), Function(V), Function(V)
#Added by Mostafa--------------------------		
Tnew, Told = Function(V), Function(V)
#------------------------------------------

E_du = ((1.0-pold)**2)*inner(grad(v),sigma(u, Told))*dx
E_phi = (Gc*l*inner(grad(p),grad(q))+((Gc/l)+2.0*H(uold,unew,Hold))\
            *inner(p,q)-2.0*H(uold,unew,Hold)*q)*dx
#Added by Mostafa--------------------------		
therm_form = (cV*(dT-Told)/deltaT*T_ + kappa*T0*tr(epsilon(unew-uold))/deltaT*T_ + dot(k*grad(dT), grad(T_)))*dx
#------------------------------------------		
p_disp = LinearVariationalProblem(lhs(E_du), rhs(E_du), unew, bc_u)
p_phi = LinearVariationalProblem(lhs(E_phi), rhs(E_phi), pnew, bc_phi)
solver_disp = LinearVariationalSolver(p_disp)
solver_phi = LinearVariationalSolver(p_phi)
#Added by Mostafa--------------------------		
p_T = LinearVariationalProblem(lhs(therm_form), rhs(therm_form), Tnew, bc_T)
solver_T = LinearVariationalSolver(p_T)
#------------------------------------------

# Initialization of the iterative procedure and output requests
t = 0
u_r = 0.007
#deltaT  = 0.1
tol = 1e-3
conc_f = File ("./ResultsDir/phi.pvd")
conc_T = File ("./ResultsDir/Temp.pvd")

fname = open('ForcevsDisp.txt', 'w')

# Staggered scheme
while t<=1.0:
    t += deltaT
    if t >=0.7:
        deltaT = deltaT #0.0001 #Edited by Mostafa
    load.t=t*u_r
    iter = 0
    err = 1

    while err > tol:
        iter += 1
        solver_disp.solve()
        solver_phi.solve()
        solver_T.solve() 

	

        err_u = errornorm(unew,uold,norm_type = 'l2',mesh = None)
        err_phi = errornorm(pnew,pold,norm_type = 'l2',mesh = None)
        err = max(err_u,err_phi)
        
        uold.assign(unew)
        pold.assign(pnew)
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
