#!/usr/bin/env python3

"""
TokaMaker Inverse Equilibrium Solver

This script solves the inverse equilibrium problem for a tokamak fusion reactor.
Given a desired plasma shape, current, and pressure, it computes the required 
coil currents by solving the Grad-Shafranov equation with appropriate constraints.

The inverse approach allows us to design a plasma configuration and then determine 
the engineering requirements (coil currents) needed to produce that configuration,
which is a key step in fusion reactor design and operation.

This script requires a pre-generated mesh file containing the geometry of the 
tokamak, including the vacuum vessel, coils, and plasma region.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Configure matplotlib for non-interactive operation (important for server environments)
import matplotlib
matplotlib.use('Agg')  # Prevents GUI window from opening
import matplotlib.pyplot as plt

# Import TokaMaker path setup utility
from simulator.tokamaker.utils import setup_tokamaker_path


def compute_inverse_equilibrium(variables, mesh_file_path, output_dir=None, myOFT=None):
    """
    Compute an inverse equilibrium solution for a tokamak plasma.
    
    This function solves the Grad-Shafranov equation to find the coil currents needed
    to create a plasma with the desired properties (shape, current, pressure).
    It uses a pre-generated mesh and configuration variables to define constraints
    for the inverse problem.
    
    Args:
        variables (dict): Dictionary of reactor parameters from variables.yaml
                          including major_radius, magnetic_field_on_axis, 
                          inverse_aspect_ratio, plasma_current, elongation, 
                          triangularity, and average_total_pressure
        mesh_file_path (str): Path to the pre-generated mesh file (.h5)
        output_dir (str, optional): Directory to save output files and visualizations
        myOFT (OFT_env, optional): Existing OpenFUSIONToolkit environment instance
    
    Returns:
        tuple: (mygs, coil_currents)
            - mygs: TokaMaker equilibrium solver object with the computed solution
            - coil_currents: Dictionary mapping coil names to currents (in Amperes)
    """
    
    # Configure OpenFUSIONToolkit Python environment
    print("Setting up TokaMaker environment...")
    if not setup_tokamaker_path():
        print("Failed to set up TokaMaker path. Aborting.")
        return None, None
    
    # Import TokaMaker modules after environment setup
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun, create_isoflux
    
    # Create output directory for results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    
    # Initialize the OpenFUSIONToolkit environment and TokaMaker solver
    if myOFT is None:
        myOFT = OFT_env(nthreads=2)  # Create with 2 computational threads
        print("Initialized OpenFUSIONToolkit environment")
    mygs = TokaMaker(myOFT)  # Create TokaMaker instance for equilibrium calculations
    
    # Load the pre-generated computational mesh
    print(f"Loading mesh from {mesh_file_path}...")
    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(mesh_file_path)
    print(f"Loaded mesh with {len(mesh_pts)} nodes and {len(mesh_lc)} elements")
    
    # Configure the TokaMaker solver with the mesh and region properties
    print("Setting up computational mesh and regions...")
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)  # Define mesh elements
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)  # Define region physical properties
    
    # Extract key plasma parameters from the variables dictionary
    print("Extracting reactor parameters from configuration...")
    major_radius = float(variables['major_radius']['value'])            # Major radius (R0) in meters
    magnetic_field = float(variables['magnetic_field_on_axis']['value']) # B0 in Tesla
    inverse_aspect_ratio = float(variables['inverse_aspect_ratio']['value']) # epsilon = a/R0
    plasma_current = float(variables['plasma_current']['value'])        # Ip in Amperes
    elongation = float(variables['elongation']['value'])                # Vertical elongation kappa
    triangularity = float(variables['triangularity']['value'])          # Triangularity delta
    
    # Calculate additional parameters
    minor_radius = major_radius * inverse_aspect_ratio  # Minor radius (a) in meters
    
    # Calculate toroidal field strength parameter F0 = B0*R0
    # This represents the vacuum toroidal field strength at the major radius
    F0 = magnetic_field * major_radius  # In Tesla*meters
    print(f"Plasma configuration: R0={major_radius:.3f}m, a={minor_radius:.3f}m, B0={magnetic_field:.2f}T, Ip={plasma_current/1e6:.2f}MA")
    print(f"Plasma shape: elongation={elongation:.2f}, triangularity={triangularity:.2f}")
    
    # Initialize the Grad-Shafranov solver
    # order=2 specifies the order of the finite element discretization
    print("Initializing Grad-Shafranov solver...")
    mygs.setup(order=2, F0=F0)
    
    # Configure vertical stability control
    # Vertical stability coils (VS) counteract plasma vertical movement
    # These are critical for elongated plasmas which are vertically unstable
    print("Setting up vertical stability control...")
    try:
        if 'VS' in mygs.coil_sets:
            # VS coil is identified in the mesh - use it for vertical stability
            mygs.set_coil_vsc({'VS': 1.0})  # Configure VS coil with unit gain
            print("Vertical stability control enabled using dedicated VS coil")
        else:
            print("No dedicated VS coil found in the mesh")
            print("Vertical stability will rely on other PF coils")
    except Exception as e:
        print(f"Warning: Could not set up vertical stability control: {e}")
        print("Simulation will continue without explicit vertical stability control")
    
    # Set operational limits on coil currents
    # These bounds constrain the optimization to physically realistic values
    print("Setting coil current limits...")
    coil_bounds = {}
    for key in mygs.coil_sets:
        if key.startswith('DIV'):
            # Divertor coils typically operate at lower currents
            coil_bounds[key] = [-1.0E6, 1.0E6]  # ±100 kA
            print(f"Setting {key} current limits to ±100 kA")
        else:
            # PF and CS coils can operate at higher currents
            coil_bounds[key] = [-20.0E6, 20.0E6]    # ±50 MA
            print(f"Setting {key} current limits to ±50 MA")
    
    # Apply the current limits to the solver
    mygs.set_coil_bounds(coil_bounds)
    
    # Set target plasma parameters for the equilibrium
    # These are the primary constraints for the inverse problem
    P0_target = float(variables['average_total_pressure']['value'])  # Average pressure in Pa
    
    print(f"Setting plasma target parameters:")
    print(f"  - Plasma current: {plasma_current/1e6:.2f} MA")
    print(f"  - Average pressure: {P0_target/1e3:.2f} kPa")
    
    # Set target values in the solver
    mygs.set_targets(Ip=plasma_current, pax=P0_target)
    
    # Define the last-closed-flux-surface (LCFS) shape using isoflux points
    print("Defining plasma boundary shape...")
    # Create points that define the plasma boundary based on shape parameters
    n_pts = 12  # Number of points along the boundary (more points = smoother shape)
    
    # Generate the plasma boundary contour using shape parameters
    # Parameters: (n_points, R0, Z0, minor_radius, elongation, triangularity)
    LCFS_contour = create_isoflux(n_pts, major_radius, 0.0, minor_radius, elongation, triangularity)
    
    # Identify potential X-point locations (useful for divertor configuration)
    # X-points occur at the top and bottom extremes of the plasma
    z_values = LCFS_contour[:, 1]
    top_idx = np.argmax(z_values)       # Highest point (potential upper X-point)
    bottom_idx = np.argmin(z_values)    # Lowest point (potential lower X-point)
    
    # Extract the coordinates for potential X-points
    top_point = LCFS_contour[top_idx].reshape(1, 2)
    bottom_point = LCFS_contour[bottom_idx].reshape(1, 2)
    
    # Display X-point locations for reference
    print(f"Potential X-point locations:")
    print(f"  - Upper: R={top_point[0,0]:.3f}m, Z={top_point[0,1]:.3f}m")
    print(f"  - Lower: R={bottom_point[0,0]:.3f}m, Z={bottom_point[0,1]:.3f}m")
    
    # Set the isoflux constraints for the boundary shape
    print("Setting flux surface constraints for the plasma boundary...")
    mygs.set_isoflux(LCFS_contour)  # All points should lie on the same flux surface
    
    # Note: X-point constraints are commented out for this calculation
    # Uncommenting would enforce X-points at the specified locations
    mygs.set_saddles(np.vstack((top_point, bottom_point)))
    
    # Configure coil current regularization
    # Regularization helps produce a more physically reasonable solution
    # by adding penalties to large or extreme coil currents
    print("Setting up coil current regularization...")
    regularization_terms = []
    
    # Check if coil definitions exist in the mesh
    # --- coil-current regularisation ----------------------------------
    reg = []
    for name in mygs.coil_sets:
        w = (2e-2 if name.startswith('CS1') else    # main CS
            1e-2 if name.startswith('CS')  else    # other CS
            5e-3 if name.startswith('PF')  else    # PF shaping
            1e-2 if name.startswith('VS')  else    # vert. stab.
            1e-3 if name.startswith('DV') else    # divertor
            1e-2)                                  # default
        reg.append(mygs.coil_reg_term({name: 1.0}, 0.0, w))

    # heavily penalise the virtual stability coil, if present
    if 'VS' in mygs.coil_sets:
        reg.append(mygs.coil_reg_term({'#VSC': 1.0}, 0.0, 1e2))

    mygs.set_coil_reg(reg_terms=reg)  

    
    # Define plasma current and pressure profiles
    # These functions describe how current and pressure vary across flux surfaces
    print("Setting up plasma current and pressure profiles...")
    
    # Create current profile function using power law form
    # Parameters: (number of points, power for (1-psi), power for psi)
    # This creates a profile with a peak near the magnetic axis that falls off toward the edge
    ffp_prof = create_power_flux_fun(40, 1.5, 2.0)  # Current profile (related to f*f')
    
    # Create pressure profile function using power law form
    # This creates a profile that falls off toward the edge
    pp_prof = create_power_flux_fun(40, 4.0, 1.0)   # Pressure profile (related to p')
    
    # Apply the profiles to the solver
    mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)
    print("Plasma profiles configured: current peaks near axis, pressure peaks at center")
    
    # Initialize the poloidal flux function (psi) with an analytical approximation
    # This provides a good starting point for the solver
    R0 = major_radius    # Major radius (m)
    Z0 = 0.0             # Vertical position of magnetic axis (m)
    a = minor_radius     # Minor radius (m)
    kappa = elongation    # Elongation
    delta = triangularity # Triangularity
    
    print("Initializing poloidal flux function...")
    print(f"Initial plasma position: (R,Z) = ({R0:.3f}m, {Z0:.3f}m)")
    print(f"Initial plasma shape: a={a:.3f}m, κ={kappa:.2f}, δ={delta:.2f}")
    
    # Initialize psi with analytical Soloviev-like solution
    mygs.init_psi(R0, Z0, a, kappa, delta)
    
    # Solve the Grad-Shafranov equation to find equilibrium
    print("\nStarting Grad-Shafranov solver for inverse equilibrium calculation...")
    try:
        # Set maximum number of iterations for better convergence
        mygs.settings.maxits = 200
        mygs.update_settings()
        
        # Run the solver (this is computationally intensive)
        print("Running solver (this may take a few minutes)...")
        result = mygs.solve()
        
        # Check result (True = success, False = failure)
        if result:
            print("Equilibrium solution converged successfully!")
        else:
            print("Solver completed but may not have fully converged")
            print("Results should be examined carefully")
    except Exception as e:
        print(f"Error in Grad-Shafranov solution: {e}")
        print("The mesh may not be compatible with the requested plasma parameters")
        print("Consider adjusting the plasma parameters or regenerating the mesh")
        # Return partial solution
        return mygs, None
    
    # Extract the computed coil currents from the solution
    print("\nExtracting final coil currents from solution...")
    try:
        coil_currents, _ = mygs.get_coil_currents()  # Returns dict of currents and virtual currents
        
        # Print summary of coil currents
        print("Solution requires the following coil currents:")
        for name, current in coil_currents.items():
            print(f"  {name}: {current/1e6:.2f} MA")
    except Exception as e:
        print(f"Warning: Could not extract coil currents: {e}")
        coil_currents = {}
    
    # Print detailed equilibrium properties
    print("\nComputed equilibrium properties:")
    try:
        mygs.print_info()  # Prints beta, safety factor, and other equilibrium properties
    except Exception as e:
        print(f"Warning: Could not print equilibrium information: {e}")
    
    # Generate visualizations if output directory is specified
    if output_dir:
        print(f"\nGenerating visualizations in {output_dir}...")
        try:
            # Create a figure for the equilibrium solution
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            # Plot the machine geometry (coils, vessel) with coil currents shown by color
            try:
                mygs.plot_machine(fig, ax, 
                                 coil_colormap='seismic',  # Red-blue colormap for currents
                                 coil_symmap=True,         # Symmetric colormap around zero
                                 coil_scale=1.0E-6,        # Scale currents to MA
                                 coil_clabel=r'$I_C$ [MA]') # Label for colorbar
                print("- Added machine geometry to plot")
            except Exception as e:
                print(f"Warning: Could not plot machine geometry: {e}")
            
            # Plot the flux surfaces (contours of constant poloidal flux)
            try:
                mygs.plot_psi(fig, ax, 
                             xpoint_color=None,     # Don't highlight X-points
                             vacuum_nlevels=4)      # Show 4 flux contours in vacuum region
                print("- Added flux surfaces to plot")
            except Exception as e:
                print(f"Warning: Could not plot flux surfaces: {e}")
            
            # Plot the constraints (isoflux points that define the plasma boundary)
            try:
                mygs.plot_constraints(fig, ax, 
                                    isoflux_color='tab:red',  # Red color for boundary points
                                    isoflux_marker='o')       # Circle markers
                print("- Added constraint points to plot")
            except Exception as e:
                print(f"Warning: Could not plot constraints: {e}")
            
            # Add labels and title
            ax.set_title('Tokamak Inverse Equilibrium Solution')
            ax.set_xlabel('R [m]')
            ax.set_ylabel('Z [m]')
            
            # Save the figure as a high-resolution PNG
            eq_plot_path = os.path.join(output_dir, 'inverse_equilibrium.png')
            plt.savefig(eq_plot_path, dpi=300)
            plt.close()
            print(f"Saved equilibrium plot to {eq_plot_path}")
        except Exception as e:
            print(f"Error creating equilibrium plot: {e}")
        
        # Save coil currents to a text file for reference
        try:
            if coil_currents:
                current_file_path = os.path.join(output_dir, 'coil_currents.txt')
                with open(current_file_path, 'w') as f:
                    f.write("Coil Currents:\n")
                    f.write("===================\n")
                    for key, current in coil_currents.items():
                        # round the current to 3 decimal place
                        current = round(current, 3)
                        f.write(f"{key:10}: {current}\n")
                print(f"Saved coil currents to {current_file_path}")
            else:
                print("No coil currents to save")
        except Exception as e:
            print(f"Error saving coil currents: {e}")

        eq_filename = os.path.join(output_dir, 'equilibrium.geqdsk') if output_dir else 'equilibrium.geqdsk'
        mygs.save_eqdsk(eq_filename, nr=200, nz=200, lcfs_pad=0.001)
        print(f"Saved equilibrium to {eq_filename}")
    
    print("\nInverse equilibrium calculation completed")
    return mygs, coil_currents


def main():
    """
    Main entry point for the inverse equilibrium calculation.
    
    Parses command line arguments, loads configuration, and runs the
    inverse equilibrium solver. Results are saved to the specified
    output directory.
    """
    import argparse
    import yaml
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Compute inverse equilibrium solution for a tokamak fusion reactor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mesh-file', type=str, default=None,
                      help='Path to the mesh file (.h5)')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory to save output files and visualizations')
    args = parser.parse_args()
    
    # Determine the project base directory
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Set default paths if not specified in arguments
    if args.mesh_file is None:
        args.mesh_file = str(base_dir / "design/YINSEN_mesh.h5")
        print(f"Using default mesh file: {args.mesh_file}")
    
    if args.output_dir is None:
        args.output_dir = str(base_dir / "results/tokamaker/equilibrium")
        print(f"Using default output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load reactor configuration from variables.yaml
    variables_file = base_dir / "design/variables.yaml"
    print(f"Loading configuration from: {variables_file}")
    
    try:
        # Load and parse the variables file
        with open(variables_file, 'r') as f:
            variables = yaml.safe_load(f)
        
        # Run the inverse equilibrium calculation
        print("\n" + "="*80)
        print("STARTING INVERSE EQUILIBRIUM CALCULATION")
        print("="*80)
        mygs, coil_currents = compute_inverse_equilibrium(variables, args.mesh_file, args.output_dir)
        
        # Report completion status
        if mygs is not None and coil_currents is not None:
            print("\n" + "="*80)
            print("INVERSE EQUILIBRIUM CALCULATION COMPLETED SUCCESSFULLY")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("INVERSE EQUILIBRIUM CALCULATION COMPLETED WITH ERRORS")
            print("="*80)
            sys.exit(1)
            
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR IN INVERSE EQUILIBRIUM CALCULATION")
        print("="*80)
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()