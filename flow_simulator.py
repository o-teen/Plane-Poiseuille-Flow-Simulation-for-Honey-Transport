"""
Plane Poiseuille Flow Simulation for Honey Transport
=========================================================
A Python implementation of the semi-implicit method for 
solving the 2-D incompressible Navier-Stokes equations on a staggered grid.

Attribution Statement
======================
Parts of the domain setup and implementation structure were adapted from an 
open-source code(https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/pipe_flow_with_inlet_and_outlet_python.py) 
(Ceyron, 2021). 
The semi-implicit solver, discretisation, and analysis presented here are
original to this project. 

Author: O-Teen Kwok
Date: 2025-12-19
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse as spy
import scipy.sparse.linalg as matrix


# ========Initialisation Functions========
def setup_domain(Ny):
    """Set up the computational domain and staggered grid"""
    
    # Indicator
    print("Setting up computational domain...")
    
    D = 0.1               # Channel width (m)
    ratio = 8             # Length-to-diameter ratio
    L = D * ratio         # Channel length (m)
    h = D / (Ny - 1)      # Grid spacing
    Nx = int(L / h + 1)   # Number of nodes in x-direction
    Nt = 160              # Number of time steps
    dt = 0.001            # Time step size (s)

    # Spatial coordinates
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, D, Ny)    
    X, Y = np.meshgrid(x, y)

    # Initialize staggered grid arrays (ghost cells included for BCs)
    u = np.zeros((Ny + 1, Nx))         # Horizontal velocity (m/s)
    v = np.zeros((Ny, Nx + 1))         # Vertical velocity (m/s)
    p = np.zeros((Ny + 1, Nx + 1))     # Pressure (Pa)

    return Nx, Nt, dt, h, x, y, X, Y, u, v, p, D


def initialise_arrays(Ny, Nx, Nt):
    """Initialise arrays to store solutions"""
    # Solution storage (no ghost cells)
    u_sol = np.zeros((Ny, Nx, Nt + 1))
    v_sol = np.zeros((Ny, Nx, Nt + 1))
    
    return u_sol, v_sol


# ========Applying Boundary Conditions and Set Initial Values========

def set_initial_conditions(u, v, p, u_sol, v_sol, uin):
    """Set initial conditions for velocity and pressure"""
    # indicator
    print("initialising...")
    
    u[1:-1, :] = uin  # u = inlet velocity everywhere except at walls
    v[:, :] = 0.0     # zero vertical velocity everywhere 
    # Pressure initialized to zero everywhere (already done in setup_domain)

    # Store initial conditions in solution arrays
    u_sol[:, :, 0] = uin
    v_sol[:, :, 0] = 0.0     

    return u, u_sol, v, v_sol, p


def apply_boundary_conditions(u, v, p, uin):
    """Apply boundary conditions to velocity and pressure."""
    # No-slip at pipe walls (Dirichlet)
    u[0, :] = -u[1, :]      # Bottom wall
    u[-1, :] = -u[-2, :]    # Top wall
    v[0, :] = 0.0           # Bottom wall
    v[-1, :] = 0.0          # Top wall
    
    # Inlet (Dirichlet)
    u[1:-1, 0] = uin
    v[:, 0] = -v[:, 1]   # v = 0 at inlet

    # Outlet (Neumann: ∂u/∂x = 0, ∂v/∂x = 0)
    u[1:-1, -1] = u[1:-1, -2]
    v[:, -1] = v[:, -2]

    """Dirichlet conditions: For u at the top and bottom walls, and v at the inlet,
    top, and bottom walls, the ghost cells values are set to the negative of the adjacent 
    cell values near the wall. This ensures the velocity u and v averages to zero at the wall,
    satisfying the no-slip condition"""

    # Pressure (Neumann: ∂p/∂n = 0 at all boundaries)
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]

    return u, v, p


# ========Numerical Solver Functions========

def compute_adv(u, v, h):
    """Compute advection terms explicitly"""
    adv_u = (
        u[1:-1, 2:] ** 2 - u[1:-1, :-2] ** 2
        ) / (4 * h) + (
            v[:-1, 1:-2] + v[:-1, 2:-1] + v[1:, 1:-2] + v[1:, 2:-1]
            ) * (
                u[2:, 1:-1] - u[:-2, 1:-1]
                ) / (8 * h)
    adv_v = (
        u[2:-1, 1:] + u[2:-1, :-1] + u[1:-2, 1:] + u[1:-2, :-1]
        ) * (
            v[1:-1, 2:] - v[1:-1, :-2]
            ) / (4 * h) + (
                v[2:, 1:-1] ** 2 - v[:-2, 1:-1] ** 2
                ) / (8 * h)
    return adv_u, adv_v


def solve_for_int_vel(u, v, p, dt, h, rho, adv_u, adv_v, Au, Av, Ny, Nx, u_int, v_int, uin):
    """Construct results vector and solve implicit diffusion for intermediate velocity"""
    b_u = (
        u[1:-1, 1:-1] - dt * (
            p[1:-1, 2:-1] - p[1:-1, 1:-2]
            ) / h / rho - dt * adv_u
        ).flatten()
    b_v = (
        v[1:-1, 1:-1] - dt * (
            p[2:-1, 1:-1] - p[1:-2, 1:-1]
            ) / h / rho - dt * adv_v
            ).flatten()
    u_int[1:-1, 1:-1] = matrix.spsolve(Au, b_u).reshape(Ny - 1, Nx - 2)
    v_int[1:-1, 1:-1] = matrix.spsolve(Av, b_v).reshape(Ny - 2, Nx - 1)
    u_int, v_int, _ = apply_boundary_conditions(u_int, v_int, p, uin=uin)
    return u_int, v_int


def compute_div(u_int, v_int, h):
    """Compute divergence"""
    div = (
        u_int[1:-1, 1:] - u_int[1:-1, :-1]
        ) / h + (
            v_int[1:, 1:-1] - v_int[:-1, 1:-1]
            ) / h
    return div


def pressure_correction(pp, h, p_iter, u_int, v_int, p, uin):
    """Solves for pressure correction using the Jacobi iteration scheme to enforce incompressibility"""
    p_cor = np.zeros_like(p)
    p_cor_prev = np.zeros_like(p)
    for _ in range(p_iter):
        p_cor[1:-1, 1:-1] = (
            p_cor_prev[1:-1, 2:] + p_cor_prev[2:, 1:-1] + p_cor_prev[1:-1, :-2] + p_cor_prev[:-2, 1:-1] - h**2 * pp
            ) / 4
        _, _, p_cor = apply_boundary_conditions(u_int, v_int, p_cor, uin=uin)
        p_cor_prev = p_cor.copy()
    return p_cor


def vel_correction(u_int, v_int, p, dt, h, rho, uin):
    """Correct velocity using the corrected pressure"""
    u_int[1:-1, 1:-1] -= dt * (
        p[1:-1, 2:-1] - p[1:-1, 1:-2]
        ) / h / rho
    v_int[1:-1, 1:-1] -= dt * (
        p[2:-1, 1:-1] - p[1:-2, 1:-1]
        ) / h / rho
    u_int, v_int, _ = apply_boundary_conditions(u_int, v_int, p, uin=uin)
    return u_int, v_int


def semi_implicit(u, v, p, Nx, Ny, Nt, dt, h, u_sol, v_sol, kv, rho, p_iter, tol):
    """Solve 2-D incompressible Navier-Stokes using semi-implicit method."""
    
    # Indicator
    print("Running semi-implicit solver...")
    print("This may take a while...")
    
    # Create arrays for intermediate velocity and pressure correction
    u_int = np.zeros_like(u)
    v_int = np.zeros_like(v)
    p_cor = np.zeros_like(p)

    # Construct diffusion matrices, with adjusted offsets to match interior domain
    coef = kv * dt / h ** 2
    Au = spy.diags([-coef, -coef, 1 + 4 * coef, -coef, -coef], [-1, -(Nx - 2), 0, 1, Nx - 2], shape=((Ny-1) * (Nx-2), (Ny-1) * (Nx-2)), format="csr")
    Av = spy.diags([-coef, -coef, 1 + 4 * coef, -coef, -coef], [-1, -(Nx - 1), 0, 1, Nx - 1], shape=((Ny-2) * (Nx-1), (Ny-2) * (Nx-1)), format="csr")

    for t in range(1, Nt + 1):
        incompressible = False               # Assume incompressibility
        while not incompressible:            # Loop until divergence ~ 0
            # Advection terms (explicit)
            adv_u, adv_v = compute_adv(u, v, h)

            # Solve the matrix equation using spsolve for intermediate velocity
            u_int, v_int = solve_for_int_vel(u, v, p, dt, h, rho, adv_u, adv_v, Au, Av, Ny, Nx, u_int, v_int, uin=0.05)

            # Pressure correction
            div = compute_div(u_int, v_int, h) # Compute divergence
            pp = div * rho / dt   # RHS of the Pressure Poisson equation
            
            p_cor = pressure_correction(pp, h, p_iter, u_int, v_int, p, uin=0.05)
            
            p += p_cor   # Update pressure
            
            # Apply boundary conditions
            _, _, p = apply_boundary_conditions(u_int, v_int, p, uin=0.05)      

            # Correct velocity using the corrected pressure
            u_int, v_int = vel_correction(u_int, v_int, p, dt, h, rho, uin=0.05)

            # Check convergence to confirm incompressibility
            if np.max(np.abs(div)) < tol:
                incompressible = True

        # Update and store solutions
        u = u_int.copy()
        v = v_int.copy()
        u_sol[:, :, t] = (u[1:, :] + u[:-1, :]) / 2
        v_sol[:, :, t] = (v[:, 1:] + v[:, :-1]) / 2

    speed = np.sqrt(u_sol ** 2 + v_sol ** 2)
    return u_sol, v_sol, speed



# ========Validation Checks========

def cfl_condition_check(uin, dt, h, kv):
    """Check Courant condition for numerical stability"""
    print("Checking Courant condition...")
    cfl = 3 / 2 * uin * dt / h  # Convective cfl
    print(f"Convective cfl: {cfl:.5f}")
    if cfl > 1:
        print("ERROR: Courant condition NOT met (CFL > 1). Terminating program.")
        exit(0)
    else:
        print("Courant condition met (CFL <= 1). Proceeding with simulation...") 


def reynolds_number_check(uin, D, kv):
    """Check Reynolds number to ensure laminar flow regime."""
    print("Checking Reynolds number...")
    Re = uin * D / kv  # Reynolds number for pipe flow
    print(f"Reynolds number: {Re:.2f}")
    if Re >= 2000:
        print("ERROR: Flow is NOT laminar (Re >= 2000). Terminating program.")
        exit(0)
    else:
        print(f"Flow is laminar (Re < 2000). Proceeding with simulation...")


# ========Plotting Results========

def anim_axial_vel_profile(u_sol, y, Nx, Ny, Nt, h, uin, neglect_outlet, fps, figsize):
    """Animate velocity profile along axial length"""
    def u_axial_profile(frame):
        ax.clear()
        ax.plot(u_sol[:, frame, Nt], y)
        ax.set_ylabel("Height y (m)")
        ax.set_xlabel("Velocity u (m/s)")
        ax.set_xlim(0, 3 / 2 * uin)
        ax.set_title(f"Velocity Profile u along Axial Length at Steady State\nDistance from Inlet: {frame * h:.2f} m")
        ax.grid()
        return ax
    
    fig, ax = plt.subplots(figsize=figsize)
    frames = range(0, Nx - neglect_outlet * 2, 2)
    u_axial_profile_anim = anim.FuncAnimation(fig, u_axial_profile, frames=frames)
    u_axial_profile_anim.save("u_profile_along_axial_length_anim.gif", fps=fps)
    print("Saved: u_profile_along_axial_length_anim.gif")


def anim_outlet_vel_evolution(u_sol, y, Nx, Ny, Nt, uin, neglect_outlet, fps, figsize):
    """Animate velocity profile evolution near outlet"""
    def u_ss_profile(frame):
        ax.clear()
        ax.plot(u_sol[:, Nx - neglect_outlet, frame], y)
        ax.set_ylabel("Height y (m)")
        ax.set_xlabel("Velocity u (m/s)")
        ax.set_xlim(0, 3 / 2 * uin)
        ax.set_title(f"Velocity u Profile Evolution Near Outlet\nTime Step: {frame}")
        ax.grid()
        return ax
    
    fig, ax = plt.subplots(figsize=figsize)
    u_near_outlet_anim = anim.FuncAnimation(fig, u_ss_profile, frames=range(0, Nt + 2, 2))
    u_near_outlet_anim.save("fully_developed_u_profile.gif", fps=fps)
    print("Saved: fully_developed_u_profile.gif")


def anim_contour_streamline_plt(u_sol, v_sol, speed, X, Y, Nx, Ny, Nt, neglect_outlet, fps, figsize):
    """Animate velocity contour and streamline evolution"""
    vmin = 0
    vmax = np.max(abs(speed))
    levels = 40
    
    def u_contour_stream(frame):
        ax.clear()
        ax.contourf(X[:, :-neglect_outlet], Y[:, :-neglect_outlet], speed[:, :-neglect_outlet, frame], 
                    levels=levels, vmin=vmin, vmax=vmax, cmap="inferno")
        ax.set_xlabel("Axial Length x (m)")
        ax.set_ylabel("Height y (m)")
        ax.streamplot(X[:, :-neglect_outlet], Y[:, :-neglect_outlet], u_sol[:, :-neglect_outlet, frame], 
                      v_sol[:, :-neglect_outlet, frame], color="black")
        ax.set_title(f"Velocity Contour and Streamline Evolution\nTime Step: {frame}")
        return ax
    
    fig, ax = plt.subplots(figsize=figsize)
    contour_stream_anim = anim.FuncAnimation(fig, u_contour_stream, frames=range(0, Nt + 2, 2))
    fig.colorbar(ax.contourf(X, Y, speed[:, :, Nt - 1], levels=levels, vmin=vmin, vmax=vmax, cmap="inferno"), ax=ax)
    contour_stream_anim.save("speed_contour_streamline_anim.gif", fps=fps)
    print("Saved: speed_contour_streamline_anim.gif")


def anim_surf_plt(u_sol, X, Y, Nx, Ny, Nt, uin, neglect_outlet, fps, figsize):
    """Animate 3D surface projection of velocity profile"""
    def u_profile_projection(frame):
        ax.clear()
        ax.set_xlabel("Axial Length x (m)")
        ax.set_ylabel("Height y (m)")
        ax.set_zlabel("Velocity u (m/s)")
        ax.set_zlim(0, 3/2 * uin)
        ax.plot_surface(X[:, :-neglect_outlet], Y[:, :-neglect_outlet], u_sol[:, :-neglect_outlet, frame], cmap="inferno")
        ax.plot_wireframe(X[:, :-neglect_outlet], Y[:, :-neglect_outlet], u_sol[:, :-neglect_outlet, frame], 
                         color="black", alpha=0.3)
        ax.set_title(f"Velocity u Profile Projection Evolution\nTime Step: {frame}")
        return ax
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    u_profile_projection_anim = anim.FuncAnimation(fig, u_profile_projection, frames=range(0, Nt + 10, 10))
    u_profile_projection_anim.save("u_profile_projection_anim.gif", fps=fps)
    print("Saved: u_profile_projection_anim.gif")


def generate_animations(u_sol, v_sol, speed, x, y, X, Y, Nx, Ny, Nt, h, uin, D):
    """Generate and save all animations."""
    
    # Indicator
    print("Generating animations...")
    
    plt.style.use("dark_background")
    
    # Neglect outlet to avoid boundary affects
    neglect_outlet = int(len(x) * 0.1) 
    fps = 10
    figsize_1 = (8, 6)  
    figsize_2 = (15, 6)
    
    anim_axial_vel_profile(u_sol, y, Nx, Ny, Nt, h, uin, neglect_outlet, fps, figsize_1)
    anim_outlet_vel_evolution(u_sol, y, Nx, Ny, Nt, uin, neglect_outlet, fps, figsize_1)
    anim_contour_streamline_plt(u_sol, v_sol, speed, X, Y, Nx, Ny, Nt, neglect_outlet, fps, figsize_2)
    anim_surf_plt(u_sol, X, Y, Nx, Ny, Nt, uin, neglect_outlet, fps, figsize_1)


def plt_ss_vel_profile(u_sol, y, Nx, Ny, Nt, uin, D, neglect_outlet, figsize):
    """Plot steady-state fully developed velocity profiles (analytical vs numerical)"""
    analytical_profile = 3/2 * uin * (1 - ((2 * y - D) / D) ** 2)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(analytical_profile, y, color="blue")
    ax.plot(u_sol[:, Nx - neglect_outlet, Nt], y, color="red")
    ax.set_xlabel("Velocity u (m/s)")
    ax.set_ylabel("Height y (m)")
    ax.set_title("Fully Developed Velocity u Profiles: Analytical vs Numerical")
    ax.grid()
    ax.legend(["Analytical", "Numerical"])
    fig.savefig("fully_developed_u_profiles_comparison.jpg")
    print("Saved: fully_developed_u_profiles_comparison.jpg")


def plt_mean_vel(u_sol, x, Nx, Ny, Nt, uin, neglect_outlet, figsize):
    """Plot mean velocity along pipe length (analytical vs numerical)"""
    uavg = np.array([np.average(abs(u_sol[:, i, Nt])) for i in range(Nx - neglect_outlet)])
    analytical_uavg = np.full_like(x[:Nx - neglect_outlet], uin)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x[:Nx - neglect_outlet], analytical_uavg, color="blue")
    ax.plot(x[:Nx - neglect_outlet], uavg, color="red")
    ax.set_ylim(0.8 * uin, 1.2 * uin)
    ax.set_xlabel("Axial Length x (m)")
    ax.set_ylabel("Mean Velocity u (m/s)")
    ax.set_title("Mean Velocity u along the Channel: Numerical vs Analytical")
    ax.legend(["Analytical", "Numerical"])
    ax.grid()
    fig.savefig("mean_u_along_pipe_length.jpg")
    print("Saved: mean_u_along_pipe_length.jpg")


def plt_vector_field(u_sol, v_sol, X, Y, Nx, Ny, Nt, D, neglect_outlet, figsize):
    """Plot steady-state velocity vector field"""
    y_plot_every = int(Ny / 10)
    x_plot_every = int(Nx / 10)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.quiver(X[::y_plot_every, :-neglect_outlet:x_plot_every], Y[::y_plot_every, :-neglect_outlet:x_plot_every], 
              u_sol[::y_plot_every, :-neglect_outlet:x_plot_every, Nt], 
              v_sol[::y_plot_every, :-neglect_outlet:x_plot_every, Nt], color="white")
    ax.set_ylim(D, 0)
    ax.set_xlabel("Axial Length x (m)")
    ax.set_ylabel("Height y (m)")
    ax.set_title("Steady-State Velocity Vector Field")
    fig.savefig("ss_velocity_vector_field.jpg")
    print("Saved: ss_velocity_vector_field.jpg")


def generate_static_plots(u_sol, v_sol, x, y, X, Y, Nx, Ny, Nt, h, uin, D):
    """Generate and save all static plots"""
    
    # Indicator
    print("Generating static plots...")
    
    plt.style.use("dark_background")
    
    # Neglect outlet to avoid boundary affects
    neglect_outlet = int(len(x) * 0.1) 
    
    figsize = (8, 6)
    figsize_quiver = (20, 6)
    
    plt_ss_vel_profile(u_sol, y, Nx, Ny, Nt, uin, D, neglect_outlet, figsize)
    plt_mean_vel(u_sol, x, Nx, Ny, Nt, uin, neglect_outlet, figsize)
    plt_vector_field(u_sol, v_sol, X, Y, Nx, Ny, Nt, D, neglect_outlet, figsize_quiver)


# ========Main========

def main():
    """Main code"""
    # Simulation parameters (Change the following)
    Ny = 40            # Number of nodes in y-direction
    kv = 0.005         # Kinematic viscosity of honey (m^2/s)
    uin = 0.05         # Uniform inlet velocity (m/s)
    tol = 0.01         # Divergence tolerance for incompressibility
    rho = 1430         # Density of honey (kg/m^3) 
    p_iter = 100       # Number of pressure correction iterations

    # Domain setup
    Nx, Nt, dt, h, x, y, X, Y, u, v, p, D = setup_domain(Ny)
    u_sol, v_sol = initialise_arrays(Ny, Nx, Nt)
    
    # Validation checks
    cfl_condition_check(uin, dt, h, kv)
    reynolds_number_check(uin, D, kv)
    
    # Set initial and boundary conditions
    u, u_sol, v, v_sol, p = set_initial_conditions(u, v, p, u_sol, v_sol, uin)
    u, v, p = apply_boundary_conditions(u, v, p, uin)
    
    # Run numerical solver
    u_sol, v_sol, speed = semi_implicit(u, v, p, Nx, Ny, Nt, dt, h, u_sol, v_sol, kv, rho, p_iter, tol)
    print("Simulation completed successfully!")
    
    # Plot results
    generate_animations(u_sol, v_sol, speed, x, y, X, Y, Nx, Ny, Nt, h, uin, D)
    generate_static_plots(u_sol, v_sol, x, y, X, Y, Nx, Ny, Nt, h, uin, D)
    print("All visualizations saved successfully!")

if __name__ == "__main__":
    main()
