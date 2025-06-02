#!/usr/bin/env python3

"""
TokaMaker Mesh Generator for Fusion Reactor Design

This script generates a computational mesh for tokamak equilibrium and stability simulations
using OpenFUSIONToolkit/TokaMaker. The mesh includes regions for plasma, vacuum, vessel, and coils.

The geometry is loaded from a JSON file and the output is saved as an HDF5 file that can be used
by other TokaMaker-based simulations.

Note: Requires h5py for HDF5 file operations (installable via pip install h5py).
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Prevents GUI window from opening
from pathlib import Path

# Configure script paths
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
base_dir = Path(os.path.dirname(os.path.dirname(script_dir)))

# Add base directory to Python path for imports
sys.path.append(str(base_dir))

# Import the utility function to set up TokaMaker Python path
from simulator.tokamaker.utils import setup_tokamaker_path

def create_mesh(variables=None, output_dir=None):
    """
    Create a computational mesh for TokaMaker simulations using reactor design parameters.
    
    The mesh is created based on geometry defined in the YINSEN_geom.json file in the design directory.
    It includes regions for plasma, vacuum vessel, coils, and surrounding air, with appropriate 
    physical properties and resolution for each region. The resulting mesh is saved both to the 
    specified output directory (if provided) and to the design directory for use by other scripts.
    
    Args:
        variables (dict, optional): Dictionary of variables from variables.yaml (not currently used)
        output_dir (str, optional): Additional directory to save output files and visualizations
        
    Returns:
        tuple: (mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict)
            - mesh_pts: Array of mesh node coordinates
            - mesh_lc: Array of mesh connectivity information
            - mesh_reg: Array of region IDs for each element
            - coil_dict: Dictionary of coil region information
            - cond_dict: Dictionary of conductor region information
    """
    # Setup TokaMaker path
    if not setup_tokamaker_path():
        return None, None, None, None, None
    
    # Import TokaMaker modules
    from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain, save_gs_mesh
    
    # Create output directory if not exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set mesh resolution (in meters) for each region
    # These values control the density of mesh elements in different regions of the tokamak
    # Higher resolution (smaller values) in regions of interest like plasma and coils
    # Lower resolution (larger values) in less critical regions like vacuum and air
    plasma_dx = 0.05  # 5 cm resolution for plasma region (fine)
    coil_dx = 0.05    # 5 cm resolution for coil regions (fine)
    vv_dx = 0.05       # 10 cm resolution for vacuum vessel regions (medium)
    vac_dx = 0.05      # 10 cm resolution for vacuum regions (medium)

    # Load geometry information
    # The geometry information (eg. bounding curves for vacuum vessels) are loaded from a JSON file.
    # For simple geometries, testing, or generative usage this can be created directly in the code.
    # However, it is often helpful to separate this information into a fixed datafile as here.
    # This JSON file contains the following:
    # - limiter: A contour of R,Z points defining the limiter (PFC) surface
    # - inner_vv: Two contours of R,Z points defining the inner and outer boundary of the inner vacuum vessel
    # - outer_vv: Two contours of R,Z points defining the inner and outer boundary of the outer vacuum vessel
    # - coils: A dictionary of R,Z,W,H values defining the PF coils as rectangles in the poloidal cross-section
    
    # Get the path to the geometry file
    geom_file = base_dir / "design/SPARC_geom.json"
    
    if not os.path.exists(geom_file):
        print(f"ERROR: Geometry file {geom_file} not found")
        return None, None, None, None, None
    
    with open(geom_file, 'r') as fid:
        geom = json.load(fid)

    # Define logical mesh regions with their physical properties
    # The tokamak is divided into several key regions, each with specific properties:
    #
    # - air:      External region outside all physical components
    # - plasma:   Central region where fusion reactions occur
    # - vacuum1:  Primary vacuum region between plasma and inner vessel wall
    # - vacuum2:  Secondary vacuum region between inner and outer vessel walls
    # - vv1:      Inner vacuum vessel wall (conducting material)
    # - vv2:      Outer vacuum vessel wall (conducting material)
    # - PF1,...:  Poloidal field coils for plasma shaping and control
    # - VS coils: Vertical stability coils (if present)

    # Create a G-S domain
    gs_mesh = gs_Domain()
    
    # Define region information for mesh with physical properties
    # Parameters:
    # 1. Region name
    # 2. Resolution (dx) in meters
    # 3. Region type (boundary, plasma, vacuum, conductor, coil)
    # 4. Additional properties depending on region type
    
    # Define main regions
    gs_mesh.define_region('air', vac_dx, 'boundary')                  # External boundary region
    gs_mesh.define_region('plasma', plasma_dx, 'plasma')              # Central plasma region
    gs_mesh.define_region('vacuum1', vv_dx, 'vacuum', allow_xpoints=True)  # Inner vacuum with X-points allowed
    gs_mesh.define_region('vacuum2', vv_dx, 'vacuum')                 # Outer vacuum region
    
    # Define conducting structures (vacuum vessel)
    # eta = resistivity in Ohm-meters (6.9E-7 is typical for stainless steel)
    gs_mesh.define_region('vv1', vv_dx, 'conductor', eta=6.9E-7)      # Inner vacuum vessel
    gs_mesh.define_region('vv2', vv_dx, 'conductor', eta=6.9E-7)      # Outer vacuum vessel
    
    # Define standard PF (Poloidal Field) coils
    for key, coil in geom['coils'].items():
        if not key.startswith('VS'):
            gs_mesh.define_region(key, coil_dx, 'coil')
    
    # Define VS (Vertical Stability) coils if present
    # These coils are paired with opposite current directions (nTurns +1.0/-1.0)
    if 'VS1U' in geom['coils'] and 'VS1L' in geom['coils']:
        gs_mesh.define_region('VS1U', coil_dx, 'coil', coil_set='VS', nTurns=1.0)   # Upper VS coil
        gs_mesh.define_region('VS1L', coil_dx, 'coil', coil_set='VS', nTurns=-1.0)  # Lower VS coil (opposite current)

    # Define geometry for region boundaries
    # Once the region types and properties are defined we now define the geometry of the mesh
    # using shapes and references to the defined regions.
    # 1. We add the limiter contour as a "polygon", referencing 'plasma' as the region enclosed
    #    by the contour and 'vacuum1' as the region outside the contour.
    # 2. We add the inner vacuum vessel as an "annulus" with curves defining the inner and outer
    #    edges respectively. We also reference 'vacuum1' as the region enclosed by the annulus,
    #    'vv1' as the annular region itself, and 'vacuum2' as the region outside the annulus.
    # 3. We add the outer vacuum vessel as an "annulus" with curves defining the inner and outer
    #    edges respectively. We also reference 'vacuum2' as the region enclosed by the annulus,
    #    'vv2' as the annular region itself, and 'air' as the region outside the annulus.
    # 4. We add each of the coils as "rectangles", which are defined by a center point (R,Z)
    #    along with a width (W) and height (H). We also reference 'air' as the region outside
    #    the rectangle for most coils and 'vacuum1' for the VS coils.

    # Define physical geometry for each region
    # Parameters for add_polygon: points, inner_name, parent_name
    # Parameters for add_annulus: inner_curve, inner_name, outer_curve, annulus_name, parent_name
    # Parameters for add_rectangle: r_center, z_center, width, height, region_name, parent_name
    
    # Define the plasma region bounded by the limiter
    gs_mesh.add_polygon(geom['limiter'], 'plasma', parent_name='vacuum1')
    
    # Define the vacuum vessel regions as annular regions
    # Inner vacuum vessel separates vacuum1 and vacuum2
    gs_mesh.add_annulus(geom['inner_vv'][0], 'vacuum1', 
                        geom['inner_vv'][1], 'vv1', 
                        parent_name='vacuum2')
    
    # Outer vacuum vessel separates vacuum2 and air
    gs_mesh.add_annulus(geom['outer_vv'][0], 'vacuum2', 
                        geom['outer_vv'][1], 'vv2', 
                        parent_name='air')
    
    # Define the poloidal field coils as rectangles
    for key, coil in geom['coils'].items():
        # VS coils are located inside the vacuum vessel
        if key.startswith('VS'):
            gs_mesh.add_rectangle(coil['rc'], coil['zc'], coil['w'], coil['h'], 
                                 key, parent_name='vacuum1')
        # Other PF coils are outside the vacuum vessel
        else:
            gs_mesh.add_rectangle(coil['rc'], coil['zc'], coil['w'], coil['h'], 
                                 key, parent_name='air')

    # Visualize the tokamak topology before mesh generation
    # This creates a plot showing the boundaries of all regions to verify geometry
    print("Plotting tokamak topology...")
    fig, ax = plt.subplots(1, 1, figsize=(4, 6), constrained_layout=True)
    gs_mesh.plot_topology(fig, ax)
    
    # Save topology plot
    if output_dir:
        topology_plot = os.path.join(output_dir, 'mesh_topology.png')
        plt.savefig(topology_plot)
        print(f"Saved topology visualization to {topology_plot}")
    else:
        plt.savefig('mesh_topology.png')
    plt.close()

    # Generate the computational mesh
    print("Generating computational mesh...")
    # The build_mesh method triangulates the geometry according to region definitions
    # This creates:
    # - mesh_pts: Node coordinates
    # - mesh_lc: Element connectivity
    # - mesh_reg: Region ID for each element
    mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()
    
    # Extract coil and conductor information for TokaMaker simulations
    coil_dict = gs_mesh.get_coils()       # Dictionary of coil properties
    cond_dict = gs_mesh.get_conductors()  # Dictionary of conductor properties
    
    # Visualize the generated mesh, color-coded by region
    print("Creating mesh visualization...")
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    gs_mesh.plot_mesh(fig, ax)
    
    # Save mesh visualization
    if output_dir:
        regions_plot = os.path.join(output_dir, 'mesh_regions.png')
        plt.savefig(regions_plot)
        print(f"Saved mesh visualization to {regions_plot}")
    else:
        plt.savefig('mesh_regions.png')
    plt.close()

    # Save the generated mesh for later use
    # Mesh generation is computationally expensive, so we save the results to HDF5 files
    # for reuse across multiple simulations. The saved mesh contains:
    # - Node coordinates (mesh_pts)
    # - Element connectivity (mesh_lc)
    # - Region IDs for each element (mesh_reg)
    # - Coil definitions (coil_dict)
    # - Conductor definitions (cond_dict)
    
    # Save to optional output directory if specified (for visualization/debugging)
    if output_dir:
        mesh_file = os.path.join(output_dir, 'YINSEN_mesh.h5')
        save_gs_mesh(mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict, mesh_file)
        print(f"Mesh saved to output directory: {mesh_file}")
    
    # Always save to the standard design directory location
    # This is the canonical location where other simulation scripts expect to find the mesh
    design_dir = base_dir / "design"
    design_mesh_file = os.path.join(design_dir, 'YINSEN_mesh.h5')
    save_gs_mesh(mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict, design_mesh_file)
    print(f"Mesh saved to design directory: {design_mesh_file}")
    
    return mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict

def main():
    """
    Main entry point for the mesh creation script.
    
    Parses command line arguments and runs the mesh creation process.
    Always saves the mesh to the design directory, and optionally to another
    specified output directory for additional visualization or analysis.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a computational mesh for TokaMaker simulations')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Additional directory to save output files (mesh is always saved to design directory)')
    args = parser.parse_args()
    
    # Run the mesh creation process with error handling
    try:
        print("Starting mesh generation process...")
        create_mesh(output_dir=args.output_dir)
        print("Mesh generation completed successfully.")
    except Exception as e:
        print(f"Error creating mesh: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()