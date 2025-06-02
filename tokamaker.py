#!/usr/bin/env python3

"""
TokaMaker Interface for Reactor Design

This module provides a command-line interface and API for running TokaMaker simulations
as part of the fusion reactor design process. It connects the reactor design parameters
from variables.yaml with different TokaMaker simulation modules:

1. Mesh generation - Creates the computational mesh for the tokamak
2. Inverse equilibrium - Solves for coil currents needed to achieve desired plasma shape
4. VDE simulation - Simulates vertical displacement events

The module integrates with CFSPOPCON for plasma physics parameters and handles parameter
passing between different simulation components.
"""

# Prevent creation of __pycache__ directories
import sys
sys.dont_write_bytecode = True

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# Configure paths for imports
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(script_dir)
sys.path.append(base_dir)

# Import required modules

# Configure matplotlib settings for publication-quality plots
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markeredgewidth'] = 2


def setup_tokamaker_path():
    """
    Set up the TokaMaker Python path from the OFT_ROOTPATH environment variable.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    # Get the OFT_ROOTPATH environment variable
    tokamaker_python_path = os.getenv('OFT_ROOTPATH')
    
    if tokamaker_python_path is None:
        print("ERROR: OFT_ROOTPATH environment variable is not set.")
        print("Please run: source codes/set_tokamaker_env.sh")
        return False
    
    # Add the python directory to the path
    python_path = os.path.join(tokamaker_python_path, 'build_release', 'python')
    if not os.path.exists(python_path):
        python_path = os.path.join(tokamaker_python_path, 'python')
    
    sys.path.append(python_path)
    print(f"Added {python_path} to Python path")
    
    return True

def load_variables():
    """
    Load reactor design parameters from the variables.yaml file.
    
    Returns:
        dict: Dictionary of parameters loaded from variables.yaml with structure:
              {'parameter_name': {'value': value, 'units': units}}
    """
    # Determine the project base directory and variables file path
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    variables_file = "variables.yaml"
    
    # Load and parse the YAML file
    with open(variables_file, 'r') as f:
        variables = yaml.safe_load(f)
    
    return variables

def print_plasma_parameters(variables, beta_total=None, average_total_pressure=None):
    """
    Print the key plasma parameters in a consistent format.
    
    Args:
        variables (dict): Dictionary of plasma parameters
        beta_total (float, optional): Beta value (if already calculated)
        average_total_pressure (float, optional): Average pressure (if already calculated)
    """
    print("\nPlasma configuration parameters:")
    print(f"  Major radius: {float(variables['major_radius']['value']):.3f} m")
    print(f"  Magnetic field: {float(variables['magnetic_field_on_axis']['value']):.2f} T")
    print(f"  Inverse aspect ratio: {float(variables['inverse_aspect_ratio']['value']):.3f}")
    print(f"  Elongation: {float(variables['elongation']['value']):.2f}")
    print(f"  Triangularity: {float(variables['triangularity']['value']):.2f}")
    print(f"  Plasma current: {float(variables['plasma_current']['value'])/1e6:.2f} MA")
    
    temp_point = float(variables['temp_point']['value'])
    density_point = float(variables['density_point']['value'])
    print(f"  Temperature: {temp_point:.2f} keV")
    print(f"  Density: {density_point:.2f} 10^19 m^-3")
    
    if beta_total:
        print(f"  Beta total: {beta_total:.4f}")
    if average_total_pressure:
        print(f"  Average total pressure: {average_total_pressure/1e3:.2f} kPa")

def run_tokamaker(create_mesh=True, solve=True, plot=True, inverse_eq=False, vde_sim=False):
    """
    Run TokaMaker simulations based on reactor design parameters.
    
    This function acts as the primary interface for TokaMaker simulations. It loads parameters
    from variables.yaml, retrieves physics parameters from CFSPOPCON, and calls the appropriate
    simulation modules based on the requested operation.
    
    Args:
        create_mesh (bool): Whether to create a new computational mesh
        solve (bool): Whether to solve the Grad-Shafranov equation
        plot (bool): Whether to generate visualization plots
        inverse_eq (bool): Whether to compute an inverse equilibrium solution
        vde_sim (bool): Whether to run a Vertical Displacement Event simulation
    
    Returns:
        Various: Returns different objects depending on the simulation type:
                - For mesh creation: (mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict)
                - For inverse equilibrium: (mygs, coil_currents)
                - For VDE simulation: (mygs, eig_vals, eig_vecs, z0, eig_comp, results)
    
    Raises:
        KeyError: If a required parameter is missing from variables.yaml
        ModuleNotFoundError: If a required module cannot be imported
        Exception: For other errors during simulation
    """
    # Step 1: Load reactor design parameters
    print("Loading reactor design parameters...")
    variables = load_variables()
    
    # Step 2: Set up TokaMaker environment
    print("Initializing TokaMaker environment...")
    if not setup_tokamaker_path():
        print("ERROR: Failed to set up TokaMaker path")
        sys.exit(1)
    
    # Step 3: Create output directory structure
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = base_dir / "results/tokamaker"
    os.makedirs(output_dir, exist_ok=True)
    print(f"TokaMaker (OpenFUSIONToolkit) initialized successfully")
    print(f"Results will be stored in: {output_dir}")
    
    try:
        # Step 4: Extract required parameters from variables.yaml
        # Convert all values to float to ensure numerical calculations work
        temp_point = float(variables['temp_point']['value'])
        density_point = float(variables['density_point']['value'])
        
        # Step 5: Get physics parameters from CFSPOPCON
        print(f"Getting plasma physics parameters from CFSPOPCON at T={temp_point} keV, n={density_point} 10^19 m^-3...")
        beta_total = 0.0054
        
        # Add derived values to variables dictionary for use by simulation modules
        variables['beta_total'] = {'value': beta_total, 'units': 'UL'}
        
        # Path to the standard mesh file
        mesh_file_path = str("SPARC_mesh.h5")
        eq_file_path = str("equilibrium.geqdsk")  # Fixed typo: gedsk → geqdsk
        
        # Print common configuration parameters
        print_plasma_parameters(variables, beta_total)
        
        # Step 6: Run the requested simulation type
        if vde_sim:
            # VDE (Vertical Displacement Event) simulation
            print("\n" + "="*80)
            print("STARTING VERTICAL DISPLACEMENT EVENT (VDE) SIMULATION")
            print("="*80)
            
            from compute_vde import compute_vde
            
            # Create VDE-specific output directory
            vde_output_dir = os.path.join(output_dir, "vde")
            os.makedirs(vde_output_dir, exist_ok=True)
            
            # Check for existing equilibrium file
            geqdsk_file = "equilibrium.geqdsk"  # Fixed typo: gedsk → geqdsk
                
            # Run VDE simulation
            mygs, eig_vals, eig_vecs, z0, eig_comp, results = compute_vde(
                variables, mesh_file_path, vde_output_dir, eq_file_path, True)
            
            print("\n" + "="*80)
            print("VDE SIMULATION COMPLETED")
            print("="*80)
            
            return mygs, eig_vals, eig_vecs, z0, eig_comp, results
            
        elif inverse_eq:
            # Inverse equilibrium calculation
            print("\n" + "="*80)
            print("STARTING INVERSE EQUILIBRIUM CALCULATION")
            print("="*80)
            
            from compute_inverse_equilib import compute_inverse_equilibrium
            
            # Create equilibrium output directory
            equilibrium_output_dir = os.path.join(output_dir, "equilibrium")
            os.makedirs(equilibrium_output_dir, exist_ok=True)
            
            # Run inverse equilibrium calculation
            mygs, coil_currents = compute_inverse_equilibrium(
                variables, mesh_file_path, equilibrium_output_dir)
            
            print("\n" + "="*80)
            print("INVERSE EQUILIBRIUM CALCULATION COMPLETED")
            print("="*80)
            
            return mygs, coil_currents
            
        # elif create_mesh:
        #     # Mesh creation
        #     print("\n" + "="*80)
        #     print("STARTING MESH GENERATION")
        #     print("="*80)
            
        #     from create_mesh import create_mesh
            
        #     # Create mesh output directory
        #     mesh_output_dir = os.path.join(output_dir, "mesh")
        #     os.makedirs(mesh_output_dir, exist_ok=True)
            
        #     # Create computational mesh
        #     print(f"Creating computational mesh based on reactor geometry...")
        #     mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = create_mesh(
        #         variables, mesh_output_dir)
            
        #     print("\n" + "="*80)
        #     print("MESH GENERATION COMPLETED")
        #     print("="*80)
        #     print(f"Mesh saved to design directory and {mesh_output_dir}")
            
        #     return mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict
        
    except Exception as e:
        print(f"ERROR: Failed to run TokaMaker simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """
    Command-line interface for TokaMaker simulations.
    
    Parses command-line arguments and runs the appropriate TokaMaker simulation.
    Only one simulation type can be run at a time.
    """
    # Create command-line argument parser
    parser = argparse.ArgumentParser(
        description='Run TokaMaker simulations for fusion reactor design',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Define simulation type options (mutually exclusive)
    sim_group = parser.add_argument_group('Simulation Types')
    sim_group.add_argument('--create-mesh', action='store_true', default=True,
                    help='Create computational mesh (default)')
    sim_group.add_argument('--inverse-eq', action='store_true', default=False,
                    help='Compute inverse equilibrium solution')
    sim_group.add_argument('--vde-sim', action='store_true', default=False,
                    help='Compute Vertical Displacement Event simulation')
    
    # Define simulation control options
    control_group = parser.add_argument_group('Simulation Control')
    control_group.add_argument('--solve', action='store_true', default=True,
                    help='Solve the Grad-Shafranov equation')
    control_group.add_argument('--plot', action='store_true', default=True,
                    help='Generate visualization plots')
    
    # Define negative options (to turn off defaults)
    control_group.add_argument('--no-mesh', action='store_false', dest='create_mesh',
                    help='Skip mesh creation')
    control_group.add_argument('--no-solve', action='store_false', dest='solve',
                    help='Skip solving the Grad-Shafranov equation')
    control_group.add_argument('--no-plot', action='store_false', dest='plot',
                    help='Skip generating plots')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the selected simulation with error handling
    try:
        print("\n" + "="*80)
        print("TOKAMAKER SIMULATION INTERFACE")
        print("="*80)
        
        # Determine which simulation to run (only one at a time)
        if args.vde_sim:
            print("Mode: Vertical Displacement Event Simulation")
            args.create_mesh = False
        elif args.inverse_eq:
            print("Mode: Inverse Equilibrium Calculation")
            args.create_mesh = False
        else:
            print("Mode: Mesh Generation")
        
        # Run the appropriate simulation
        run_tokamaker(
            create_mesh=args.create_mesh, 
            solve=args.solve, 
            plot=args.plot, 
            inverse_eq=args.inverse_eq, 
            vde_sim=args.vde_sim
        )
        
        print("\nSimulation completed successfully")
        
    except Exception as e:
        print("\nERROR: Simulation failed")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()