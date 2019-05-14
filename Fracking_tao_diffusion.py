# FEnics code  Variational Fracture Mechanics Hi Hi
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

from math import hypot, atan2,erfc
#=======================================================================================
def vec(z):
    if isinstance(z, dolfin.cpp.Function):
        return dolfin.as_backend_type(z.vector()).vec()
    else:
       return dolfin.as_backend_type(z).vec()

def mat(A):
        return dolfin.as_backend_type(A).mat()

#=======================================================================================
# Setup and parameters
#=======================================================================================
set_log_level(20)

# parameters of the nonlinear solver used for the alpha-problem
g6solver_alpha_parameters={"method", "tron", 			# when using gpcg make sure that you have a constant Hessian
               "monitor_convergence", True,
                       #"line_search", "gpcg"
                       #"line_search", "nash"
                       #"preconditioner", "ml_amg"
               "report", True}

#=======================================================================================
# Fracking Function
#=======================================================================================
def Fracking(E, nu, k, mu_dynamic):

    #=======================================================================================
    # Input date
    #=======================================================================================
    # Geometry
    L = 2.5 # m,length
    H = 1. # m, height
    Lx= 50
    Ly= 10

    meshname="fracking_hsize%g" % (Lx)

    # Material constants
    PlaneStress= False


    # Stopping criteria for the alternate minimization
    max_iterations = 200
    tolerance = 1.0e-6

    # Loading
    body_force = Constant((0.,0.))  # bulk load
    pressure_min = 0. # load multiplier min value
    pressure_max = 1. # load multiplier max value
    pressure_steps = 600# number of time steps

    #====================================================================================
    # To define  pressure field
    #====================================================================================
    # Input data: Table 3.2 of Chukwudozie 2016, page 104

    biot=1.0         # Biot coefficient
    kappa= 1.        #is the permeability of the rock #m^2


    nu_u = 0.5       #undrained Poisson ratio	
    B=1.0            #the Skempton coefficient
    F=2.5e6          #Pa.m=N/m, Load, The application of a load (2F) causes an instantaneous and uniform pressure increase throughout the domain
    G=E/(2.*(1.+nu))
    c=2.*kappa*B**2.*G*(1.-nu)*(1.+nu_u)**2/(9.*mu_dynamic*(1-nu_u)*(nu_u-nu))*1e-6
    print(c/L**2)

    DeltaT = 0.01  
    #=======================================================================================
    # Geometry and mesh generation
    #=======================================================================================
    mesh = RectangleMesh(Point(0., 0.), Point(L, H), Lx, Ly)
    plt.figure(1)
    plot(mesh, "2D mesh")
    plt.interactive(True)
    ndim = mesh.geometry().dim() # get number of space dimensions
    zero_v = Constant((0.,)*ndim)

    #=======================================================================================
    # strain, stress and strain energy for Isotropic and Amor's model
    #=======================================================================================
    def eps(u_):
        """
        Geometrical strain
        """
        return sym(grad(u_))

    #----------------------------------------------------------------------------------------
    def sigma0(u_):
        """
        Application of the sound elasticy tensor on the strain tensor
        """
        Id = Identity(len(u_))
        return 2.0*mu*eps(u_) + lmbda*tr(eps(u_))*Id
    
    #----------------------------------------------------------------------------------------
    def psi_0(u_):
        """
        The strain energy density for a linear isotropic ma-
        terial
        """
        return  0.5 * lmbda * tr(eps(u_))**2 + mu * eps(u_)**2


    #=======================================================================================
    # others definitions
    #=======================================================================================
    prefix = "L%s-H%.2f-steps%s,mu%s"%(L,H, pressure_steps,k)
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

    # Neumann boundary condition
    class MyExpr(UserExpression):
            def eval(self, values, x):

                if abs(x[1] - 1.)<DOLFIN_EPS:
                    L = 2.5 # m,length
                    H = 1. # m, height
                    biot=1.0         # Biot coefficient
                    kappa= 1.        #is the permeability of the rock #m^2
                    nu_u = 0.5       #undrained Poisson ratio
                    B=1.0            #the Skempton coefficient
                    F=2.5e6          #Pa.m=N/m, Load, The application of a load (2F) causes an instantaneous and uniform pressure increase throughout the domain
                    G=E/(2.*(1.+nu))
    
                    values[0]= 0.0
                    values[1] = -H/(4.*G*L)*(2.*(1.-nu)*F+biot*(1.-2.*nu)*assemble(P_*ds(3)))

            def value_shape(self):
                return (2,)

    my_expr = MyExpr(degree=1)


    # Initialize sub-domain instances
    right=Right()
    left=Left()
    top=Top()
    bottom=Bottom()

    # define meshfunction to identify boundaries by numbers
    # boundaries = FacetFunction("size_t", mesh)
    boundaries = MeshFunction('size_t', mesh, 1, mesh.domains())
    boundaries.set_all(9999)
    right.mark(boundaries, 1) # mark top as 1
    left.mark(boundaries, 2) # mark top as 2
    top.mark(boundaries, 3) # mark top as 3
    bottom.mark(boundaries, 4) # mark bottom as 4



    # Define new measure including boundary naming 
    # ds = Measure("ds")[boundaries] # left: ds(1), right: ds(2)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)  # left: ds(1), right: ds(2)
    #=======================================================================================
    # Variational formulation
    #=======================================================================================
    # Create function space for 2D elasticity + Damage
    V_u = VectorFunctionSpace(mesh, "CG", 1)
    V_p = FunctionSpace(mesh, "CG", 1)

    # Define the function, test and trial fields
    u_, u, u_t = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
    P_, P, P_t= Function(V_p), TrialFunction(V_p), TestFunction(V_p),


    P_0 = interpolate(Expression("F*B*(1+nu_u)/(3*L)", F=F,nu_u=nu_u, B=B, L=L, degree=1), V_p) #intial pressure  (eq41, Andro Mikelic  Bin Wang  Mary F Wheeler2014)
    u_0 =interpolate(Expression(('F*nu_u/(2*G)*x[0]/H', '-F*(1-nu_u)*H/(2*G*L)*x[1]/L'),F=F,nu_u=nu_u, G=G, L=L, H=H, degree=1), V_u) # intial disp (eq42-43, Andro Mikelic  Bin Wang  Mary F Wheeler2014)
    #=======================================================================================
    # Dirichlet boundary condition for a traction test boundary
    #=======================================================================================
    ## bc - u (imposed displacement)
    Gamma_u_0 = DirichletBC(V_u, zero_v, boundaries, 4)
    Gamma_u_1 = DirichletBC(V_u, my_expr, boundaries, 3)
    #Gamma_u_1= DirichletBC(V_u.sub(1), 0.0, bottom, method="pointwise") #the bottom side in y direction is fixed
    bc_u =[Gamma_u_0, Gamma_u_1]

    ## bc - P (imposed pressure)
    Gamma_P_0 = DirichletBC(V_p, 0.0, boundaries, 1)
    Gamma_P_1 = DirichletBC(V_p, 0.0, boundaries, 2)
    # bc_P = [Gamma_P_0, Gamma_P_1]
    bc_P = [Gamma_P_1]
    #====================================================================================
    # Define  problem and solvers
    #====================================================================================
    Pressure = (div(u_)*P*dx- div(u_0)*P*dx)+P_*P*dx-P_0*P*dx + DeltaT*(kappa/mu_dynamic)*inner(nabla_grad(P_), nabla_grad(P))*dx #check Chukwudozie 2016, page 76 eq:3.16. M=inf, alpha=1, kappa=1 (I suppose)
    #------------------------------------------------------------------------------------
    elastic_energy = psi_0(u_)*dx-biot*P_*div(u_)*dx+dot(Constant((0.,0.1e20)), u_t)*ds(3) #eq (1.5)paper :PHASE-FIELD MODELING OF PRESSURIZED FRACTURES IN A POROELASTIC MEDIUM, ANDRO MIKELIC et al.
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
    results = []
    load_multipliers = np.linspace(pressure_min,pressure_max,pressure_steps)
    file_u = File(save_dir+"/u.pvd") # use .pvd if .xdmf is not working
    file_p = File(save_dir+"/p.pvd") 
    file_ux= File(save_dir+"/ux.pvd") 
    file_uy= File(save_dir+"/uy.pvd") 


    fig, ax = plt.subplots(1, 1)
    pressure = np.zeros((Lx//2+1,1))
    xcoord = np.zeros((Lx//2+1,1))
    colors_i = ['r','b','g','m','c','k']
    colorID=0

    pressure_center= np.zeros((len(load_multipliers),2))
    dispVertical_center= np.zeros((len(load_multipliers),2))

    #=======================================================================================
    # Solving at each timestep
    #=======================================================================================
    for (i_t, t) in enumerate(load_multipliers):

        print("\033[1;32m--- Time step %d: time = %g---\033[1;m" % (i_t, c/L**2*i_t*DeltaT))


        solver_pressure.solve()

        # solve elastic problem
        solver_u.solve()

        my_expr = MyExpr(degree=1)


        u_0.vector()[:] = u_.vector()
        P_0.vector()[:] = P_.vector()

        ux = project(u_[0], V_p)
        uy = project(u_[1], V_p)

        
        # Dump solution to file
        file_u << u_
        file_p << P_ 
        file_ux << ux
        file_uy << uy


        #-----------------------------------------------------------------
        p_nodal_values = P_.vector()
        p_array = p_nodal_values.get_local()
        coor = mesh.coordinates()
        n=0
        for i in range(len(p_array)):
            if abs(coor[i][1] - H/2) < DOLFIN_EPS and coor[i][0]> L - DOLFIN_EPS:
                     #print 'p(%8g,%8g) = %g' % (coor[i][0], coor[i][1], P_(coor[i][0], coor[i][1])/1e6)
                 pressure[n] = P_(coor[i][0], coor[i][1])/0.5e6
                 xcoord[n] = coor[i][0]
                 n=n+1

            if abs(coor[i][1] - H/2) < DOLFIN_EPS and abs(coor[i][0] - L)< DOLFIN_EPS:
                     #print 'p_CENTER(%8g,%8g) = %g' % (coor[i][0], coor[i][1], P_(coor[i][0], coor[i][1])/1e6)
                 #print u_(coor[i][0], coor[i][1])[0]
                 pressure_center[i_t] = np.array([i_t*DeltaT, P_(coor[i][0], coor[i][1])/1e6]) 
                 dispVertical_center[i_t] = np.array([i_t*DeltaT, u_(coor[i][0], coor[i][1])[1]]) 



        if i_t==0 or i_t==10 or  i_t==200 or i_t==300 or i_t==599:
                     ax.plot(xcoord-L, pressure, '-', dashes=[8, 4, 2, 4, 2, 4], color=colors_i[colorID])
                     plt.grid(True)
                     plt.xlabel("x")
                     plt.ylabel("pressure (MPa)")
                     fig.suptitle("pressure profile" , fontsize=14)
                     colorID=colorID+1
        #-----------------------------------------------------------------




    plt.savefig(save_dir+"Pressure", format="jpeg")


    fig, ax_p = plt.subplots(1, 1)
    ax_p.plot(pressure_center[:,0], pressure_center[:,1],'b')
    plt.grid(True)
    plt.xlabel("$t$")
    plt.ylabel("pressure (MPa)")
    fig.suptitle("pressure of domain center " , fontsize=14)
    plt.savefig(save_dir+"CenterPressure", format="jpeg")

    fig, ax_u = plt.subplots(1, 1)
    ax_u.plot(dispVertical_center[:,0], dispVertical_center[:,1],'-r')
    plt.grid(True)
    plt.xlabel("$t$")
    plt.ylabel("vertical displacement (m)")
    fig.suptitle("vertical displacement of domain center " , fontsize=14)
    plt.savefig(save_dir+"dispVertical_center", format="jpeg")






if __name__ == '__main__':
        Fracking()

