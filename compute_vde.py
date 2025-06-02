#!/usr/bin/env python3

"""
Compute VDE (Vertical Displacement Event) using TokaMaker (OpenFUSIONToolkit)

This script performs VDE simulation for a tokamak plasma using OpenFUSIONToolkit/TokaMaker.
It uses an existing equilibrium (from inverse_equilibrium computation) as starting point and
simulates vertical instability growth and displacement.
"""

import os
import sys
import numpy as np
# Set matplotlib to use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Prevents GUI window from opening
import matplotlib.pyplot as plt
from pathlib import Path
import time
import argparse
import yaml

# Add the parent directory to the Python path so we can import from interfaces
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
base_dir = Path(os.path.dirname(os.path.dirname(script_dir)))
sys.path.append(str(base_dir))
from simulator.tokamaker.utils import setup_tokamaker_path
mygs = None  # Global variable to hold the TokaMaker instance


def compute_vde(variables, mesh_file_path, output_dir=None, eq_file_path=None, use_saved_eq=False):
    """
    Compute VDE simulation for a tokamak plasma.
    
    Args:
        variables (dict): Dictionary of variables from variables.yaml
        mesh_file_path (str): Path to the mesh file (.h5)
        output_dir (str, optional): Directory to save output files
        eq_file_path (str, optional): Path to the GEQDSK file from a previous run
        use_saved_eq (bool, optional): Whether to use a saved equilibrium file
    
    Returns:
        tuple: Various simulation results
    """
    
    # Setup TokaMaker path
    if not setup_tokamaker_path():
        return None, None, None

    # Import TokaMaker modules
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun, create_isoflux, read_eqdsk
    
    # Create output directory if not exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    myOFT = OFT_env(nthreads=2)
    mygs = TokaMaker(myOFT)

    # Load the pre-generated computational mesh
    print(f"Loading mesh from {mesh_file_path}...")
    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(mesh_file_path)
    print(f"Loaded mesh with {len(mesh_pts)} nodes and {len(mesh_lc)} elements")
    
    # Configure the TokaMaker solver with the mesh and region properties
    print("Setting up computational mesh and regions...")
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)  # Define mesh elements
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)  # Define region physical properties
    
    print("Initializing Grad-Shafranov solver...")
    F0 =  float(variables['magnetic_field_on_axis']['value']) *  float(variables['major_radius']['value'])  # In Tesla*meters
    mygs.setup(order=2, F0=F0)
    
    print(f"Loading equilibrium from GEQDSK file: {eq_file_path}")
    eqdsk = read_eqdsk(eq_file_path)
    
    print ("Setting up initial equilibrium...")

    print("Successfully set flux profiles from GEQDSK file")

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
            coil_bounds[key] = [-20.0E6, 20.0E6]    # ±10 MA
            print(f"Setting {key} current limits to ±20 MA")
    
    # Apply the current limits to the solver
    mygs.set_coil_bounds(coil_bounds)


    major_radius = float(variables['major_radius']['value'])  # Plasma major radius
    elongation = float(variables['elongation']['value'])  # Plasma elongation
    triangularity = float(variables['triangularity']['value'])  # Plasma triangularity
    inverse_aspect_ratio = float(variables['inverse_aspect_ratio']['value'])  # Inverse aspect ratio
    plasma_current = float(variables['plasma_current']['value']) # Plasma current in Amperes
    minor_radius = major_radius * inverse_aspect_ratio  # Plasma minor radius
    beta_target = float(variables['beta_total']['value'])  # Target beta value
    
    # Set target plasma parameters for the equilibrium
    # These are the primary constraints for the inverse problem
    # P0_target = float(variables['average_total_pressure']['value'])  # Average pressure in Pa
    
    print(f"Setting plasma target parameters:")
    print(f"  - Plasma current: {plasma_current/1e6:.2f} MA")
    # print(f"  - Average pressure: {P0_target/1e3:.2f} kPa")
    
    # Set target values in the solver
    mygs.set_targets(Ip=plasma_current, pax=eqdsk['pres'][0])
    
    # Define the last-closed-flux-surface (LCFS) shape using isoflux points
    print("Defining plasma boundary shape...")
    isoflux_pts = eqdsk['rzout'].copy()
    mygs.set_isoflux(isoflux_pts)
    
    # Configure coil current regularization
    # Regularization helps produce a more physically reasonable solution
    # by adding penalties to large or extreme coil currents
    print("Setting up coil current regularization...")
    regularization_terms = []
    
    # Check if coil definitions exist in the mesh
    # --- coil-current regularisation ----------------------------------
    for name in mygs.coil_sets:
        w = (2e-2 if name.startswith('CS1') else    # main CS
            1e-2 if name.startswith('CS')  else    # other CS
            5e-3 if name.startswith('PF')  else    # PF shaping
            1e-2 if name.startswith('VS')  else    # vert. stab.
            1e-3 if name.startswith('DV') else    # divertor
            1e-2)                                  # default
        regularization_terms.append(mygs.coil_reg_term({name: 1.0}, 0.0, w))

    # heavily penalise the virtual stability coil, if present
    if 'VS' in mygs.coil_sets:
        regularization_terms.append(mygs.coil_reg_term({'#VSC': 1.0}, 0.0, 1e2))

    mygs.set_coil_reg(reg_terms=regularization_terms)  

        # Set flux functions from GEQDSK file with proper formatting
    print("Formatting flux functions from GEQDSK file")
    
    # Extract flux functions from GEQDSK
    ffprim = eqdsk['ffprim']
    pprime = eqdsk['pprime']


    psi_eqdsk = np.linspace(0.0,1.0,np.size(ffprim))
    psi_sample = np.linspace(0.0,1.0,50)

    psi_prof = np.copy(psi_sample)
    ffp_prof = np.transpose(np.vstack((psi_prof,np.interp(psi_sample,psi_eqdsk,-ffprim)))).copy()
    pp_prof = np.transpose(np.vstack((psi_prof,np.interp(psi_sample,psi_eqdsk,-pprime)))).copy()

    mygs.set_profiles(ffp_prof={'type': 'linterp', 'y': ffp_prof[:,1], 'x': psi_sample},pp_prof={'type': 'linterp', 'y': pp_prof[:,1], 'x': psi_sample})

    # Initialize psi for the equilibrium

    print("Initializing psi for the equilibrium...")
    R0 = eqdsk['rcentr']
    Z0 = eqdsk['zaxis']

    a     = major_radius * inverse_aspect_ratio # Plasma minor radius
    kappa = elongation  # Plasma elongation
    delta = triangularity  # Plasma triangularity

    print(f"R0: {R0}, Z0: {Z0}, a: {a}, kappa: {kappa}, delta: {delta}")
    
    mygs.init_psi(R0, Z0, a, kappa, delta)

    # Re-solve equilibrium with updated settings
    print("Solving the equilibrium with updated settings...")

    mygs.solve()

    # Plot the equilibrium
    fig, ax = plt.subplots(figsize=(8, 8))
    mygs.plot_machine(fig, ax, coil_colormap='seismic', coil_symmap=True, 
                     coil_scale=1.E-6, coil_clabel=r'$I_C$ [MA]')
    mygs.plot_psi(fig, ax, vacuum_nlevels=4)
    ax.set_title('Initial Equilibrium')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'initial_equilibrium.png'), dpi=300)
    else:
        plt.savefig('initial_equilibrium.png', dpi=300)
    plt.close()

    psi,f,fp,p,pp = mygs.get_profiles()
    psi_q,qvals,ravgs,dl,rbounds,zbounds = mygs.get_q(psi_pad=0.01)
    fig, ax = plt.subplots(3,1,sharex=True)
    psi_eqdsk = np.linspace(0.0,1.0,eqdsk['nr'])
    psi_sample = np.linspace(0.0,1.0,10)
    # Plot F*F'
    ax[0].plot(psi,f*fp)
    ax[0].plot(psi_eqdsk,eqdsk['ffprim'],'+')
    ax[0].set_ylabel("FF'")
    # Plot P'
    ax[1].plot(psi,pp)
    ax[1].plot(psi_eqdsk,eqdsk['pprime'],'+')
    ax[1].set_ylabel("P'")
    # Plot q
    ax[2].plot(psi_q,qvals)
    ax[2].plot(psi_eqdsk,eqdsk['qpsi'],'+')
    ax[2].set_ylabel("q")
    _ = ax[-1].set_xlabel(r"$\hat{\psi}$")

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'profiles.png'), dpi=300)
    else:
        plt.savefig('profiles.png', dpi=300)
    plt.close()

    # Modify profiles to remove edge current
    edge_ramp = (1.0-ffp_prof[:,0])/(0.15)
    edge_ramp[edge_ramp > 1.0] = 1.0
    mygs.set_profiles(ffp_prof={'type': 'linterp', 'y': edge_ramp*ffp_prof[:,1], 'x': ffp_prof[:,0]},
                    pp_prof={'type': 'linterp', 'y': edge_ramp*pp_prof[:,1], 'x': pp_prof[:,0]})
    # Update global targets
    mygs.set_targets(Ip=eqdsk['ip'],Ip_ratio=(1.0/beta_target - 1.0),V0=eqdsk['zaxis'])
    mygs.solve()

    psi,f,fp,p,pp = mygs.get_profiles()
    psi_q,qvals,ravgs,dl,rbounds,zbounds = mygs.get_q(psi_pad=0.01)
    fig, ax = plt.subplots(3,1,sharex=True)
    psi_eqdsk = np.linspace(0.0,1.0,eqdsk['nr'])
    psi_sample = np.linspace(0.0,1.0,10)
    # Plot F*F'
    ax[0].plot(psi,f*fp)
    ax[0].plot(psi_eqdsk,eqdsk['ffprim'],'+')
    ax[0].set_ylabel("FF'")
    # Plot P'
    ax[1].plot(psi,pp)
    ax[1].plot(psi_eqdsk,eqdsk['pprime'],'+')
    ax[1].set_ylabel("P'")
    # Plot q
    ax[2].plot(psi_q,qvals)
    ax[2].plot(psi_eqdsk,eqdsk['qpsi'],'+')
    ax[2].set_ylabel("q")
    _ = ax[-1].set_xlabel(r"$\hat{\psi}$")

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'profiles_after_edge_ramp.png'), dpi=300)
    else:
        plt.savefig('profiles_after_edge_ramp.png', dpi=300)
    plt.close()
    

    # Print equilibrium information
    mygs.print_info()
   
    # Step 2: Compute linear stability (growth rates)
    print("Computing linear stability...")
    eig_vals, eig_vecs = mygs.eig_td(-5.E3, 10, False)
    
    # Select mode index for the eigenvector to use - Change this value to select a different mode
    mode_index = 0
    
    # Plot the most unstable mode (usually vertical displacement mode)
    fig, ax = plt.subplots(figsize=(8, 8))
    mygs.plot_machine(fig, ax, limiter_color=None)
    # Use the specified mode index for the eigenvector
    unstable_mode = eig_vecs[mode_index, :]
    mygs.plot_psi(fig, ax, unstable_mode, normalized=False, xpoint_color=None, opoint_color=None)
    mygs.plot_eddy(fig, ax, dpsi_dt=unstable_mode * abs(eig_vals[mode_index, 0]))
    ax.set_title(f'Most Unstable Mode (Eigenvector {mode_index})')
    
    print(f'Growth rate = {-eig_vals[mode_index,0]:.4E} [s^-1]')
    print(f'Growth time = {-1.0/eig_vals[mode_index,0]:.4E} [s]')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'unstable_mode.png'), dpi=300)
    else:
        plt.savefig('unstable_mode.png', dpi=300)
    plt.close()
    
    # Step 3: Initialize VDE simulation
    # Create a small initial displacement in the direction of the unstable mode
    psi0 = mygs.get_psi(normalized = False)
    max_loc = abs(eig_vecs[0,:]).argmax()
    psi_ic = psi0-eig_vecs[0,:]*(mygs.psi_bounds[1]-mygs.psi_bounds[0])/eig_vecs[0,max_loc]/ 1000  # Initial displacement of 10% of the mode amplitude

    fig, ax = plt.subplots(1,1)
    mygs.plot_machine(fig,ax,limiter_color=None)
    mygs.plot_psi(fig,ax,(psi_ic-mygs.psi_bounds[1])/(mygs.psi_bounds[0]-mygs.psi_bounds[1]),xpoint_color=None,opoint_color=None)
    mygs.set_psi(psi_ic)

    #save image
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'initial_displacement.png'), dpi=300)
    else:
        plt.savefig('initial_displacement.png', dpi=300)
    plt.close()
        
    # Step 4: Run the time-dependent VDE simulation
    # ────────── Step 4: Run the time-dependent VDE simulation ──────────
    print("Starting VDE simulation…")
    mygs.settings.pm=False
    mygs.update_settings()
    mygs.set_isoflux(None)
    mygs.set_targets(Ip=eqdsk['ip'],Ip_ratio=(1.0/beta_target - 1.0))
    mygs.setup_td(1.E-03,1.E-13,1.E-11,pre_plasma=False)

    import time
    sim_time = 0.0
    dt = 1.E-3/-eig_vals[0,0]
    print(f"Simulation time step: {dt:.2E} seconds")
    nplot = 1
    nstatus = 1
    results = [mygs.get_psi()]
    results_raw = [mygs.get_psi(False)]
    z0 = [[sim_time,mygs.o_point[1]],]
    eig_comp = [[sim_time,np.dot(eig_vecs[0,:],mygs.get_psi(normalized=False)-psi0)],]
    t00 = time.perf_counter()

    for i in range(40):
        t0 = time.perf_counter()
        sim_time, _, nl_its, lin_its, nretry = mygs.step_td(sim_time,dt)
        t1 = time.perf_counter()
        if i % nstatus == 0:
            print('{0:.5E} {1:.5E} {2:4d} {3:4d} {5:8.3f} {4:4d}'.format(sim_time,dt,nl_its,lin_its,nretry,t1-t0))
        z0.append([sim_time,mygs.o_point[1]])
        eig_comp.append([sim_time,np.dot(eig_vecs[0,:],mygs.get_psi(normalized=False)-psi0)])
        if i % nplot == 0:
            results.append(mygs.get_psi())
            results_raw.append(mygs.get_psi(False))
    t1 = time.perf_counter()
    print('Total time = {0:8.3f}'.format(t1-t00))

    
    # Step 5: Plot results of the VDE simulation
    
    # Plot evolution of flux surfaces
    import matplotlib as mpl
    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 6))
    mygs.plot_machine(fig, ax[0])
    colors = plt.cm.jet(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        mygs.plot_psi(fig, ax[0], psi=result, plasma_nlevels=1, plasma_color=[colors[i]], 
                     vacuum_nlevels=0, xpoint_color=None, opoint_color=None)
    
    norm = mpl.colors.Normalize(vmin=0.0, vmax=sim_time*1.E3)
    mygs.plot_machine(fig, ax[1])
    
    for i, result in enumerate(results):
        mygs.plot_psi(fig, ax[1], psi=result, plasma_nlevels=1, plasma_color=[colors[i]], 
                     vacuum_nlevels=0, xpoint_color=None, opoint_color=None)
    
    # Show limiter point if available
    try:
        ax[1].plot(mygs.lim_point[0], mygs.lim_point[1], 'ro')
    except:
        pass
    
    # Show mesh for the plasma region
    try:
        ax[1].triplot(mesh_pts[:, 0], mesh_pts[:, 1], mesh_lc[mesh_reg == 3, :], lw=0.5)
    except:
        pass
    
    norm = mpl.colors.Normalize(vmin=0.0, vmax=sim_time*1.E3)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet), ax=ax[1], label='Time [ms]')
    ax[0].set_title('Global VDE Evolution')
    ax[1].set_title('VDE Evolution Detail')
    
    # Zoom in to see details of the crash
    try:
        z_min = min([z[1] for z in z0])
        ax[1].set_ylim(z_min - 0.3, z_min + 0.2)
        # Set x limits around the o-point
        r_opoint = mygs.o_point[0]
        ax[1].set_xlim(r_opoint - 0.3, r_opoint + 0.3)
    except:
        # If setting limits fails, just use a reasonable range
        ax[1].set_ylim(-1.3, -0.5)
        ax[1].set_xlim(1.6, 2.1)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'vde_evolution.png'), dpi=300)
    else:
        plt.savefig('vde_evolution.png', dpi=300)
    plt.close()
    
    # Plot time-evolution of Z position and mode amplitude
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    z_hist = np.asarray(z0)
    z_hist = z_hist[1:, :] - [z_hist[1, 0], z_hist[0, 1]]  # Normalize to initial position
    
    ax[0].semilogy(z_hist[:, 0]*1.E3, abs(z_hist[:, 1]))
    ax[0].semilogy(z_hist[:20, 0]*1.E3, abs(z_hist[0, 1])*np.exp(-z_hist[:20, 0]*eig_vals[mode_index, 0]), 'r--', 
                  label=f'Linear growth rate: {-eig_vals[mode_index,0]:.1f} s⁻¹')
    ax[0].set_ylabel(r'$|Z_0 - Z_{0,initial}|$ [m]')
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_title('Vertical Position Displacement')
    
    eig_hist = np.asarray(eig_comp)
    eig_hist = eig_hist[1:, :]
    
    ax[1].semilogy(eig_hist[:, 0]*1.E3, abs(eig_hist[:, 1]))
    ax[1].semilogy(eig_hist[:20, 0]*1.E3, abs(eig_hist[0, 1])*np.exp(-eig_hist[:20, 0]*eig_vals[mode_index, 0]), 'r--', 
                  label=f'Linear growth rate: {-eig_vals[mode_index,0]:.1f} s⁻¹')
    ax[1].set_ylabel(r'$|\Delta \psi|$ [Wb]')
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_title('Unstable Mode Amplitude')
    
    ax[-1].set_xlabel(r'Time [ms]')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'vde_time_evolution.png'), dpi=300)
    else:
        plt.savefig('vde_time_evolution.png', dpi=300)
    plt.close()
    
    # Create an animation of the VDE if matplotlib animation is available
    try:
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=(8, 8))
        mygs.plot_machine(fig, ax)
        
        # First frame with initial plasma
        psi_contour = mygs.plot_psi(fig, ax, psi=results[0], plasma_nlevels=1, plasma_color=['r'], 
                                   vacuum_nlevels=0, xpoint_color=None, opoint_color=None)
        
        def update(frame):
            # Clear previous contours
            for coll in ax.collections:
                if coll not in ax.collections[:3]:  # Keep machine components
                    coll.remove()
            
            # Plot new contours
            mygs.plot_psi(fig, ax, psi=results[frame], plasma_nlevels=1, plasma_color=['r'], 
                         vacuum_nlevels=0, xpoint_color=None, opoint_color=None)
            ax.set_title(f'Time: {z0[frame][0]*1e3:.2f} ms')
            
        anim = FuncAnimation(fig, update, frames=len(results), interval=200)
        
        if output_dir:
            anim_path = os.path.join(output_dir, 'vde_animation.gif')
            anim.save(anim_path, writer='pillow', dpi=100)
        else:
            anim_path = 'vde_animation.gif'
            anim.save(anim_path, writer='pillow', dpi=100)
        plt.close()
        print(f"Created VDE animation: {anim_path}")
    except Exception as e:
        print(f"Could not create animation: {e}")
        
    
    return mygs, eig_vals, eig_vecs, z0, eig_comp, results

def main():
    """Main function to run VDE simulation."""
    parser = argparse.ArgumentParser(description='Compute VDE simulation for a tokamak plasma')
    parser.add_argument('--mesh-file', type=str, default=None,
                       help='Path to the mesh file (.h5)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save output files')
    parser.add_argument('--eq-file', type=str, default=None,
                       help='Path to a GEQDSK equilibrium file')
    parser.add_argument('--use-saved-eq', action='store_true', default=False,
                       help='Use saved equilibrium instead of computing a new one')
    args = parser.parse_args()
    
    # Get the base directory
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Set default mesh file if not specified
    if args.mesh_file is None:
        args.mesh_file = str(base_dir / "design/YINSEN_mesh.h5")
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = str(base_dir / "results/tokamaker/vde")
    
    # Load variables from variables.yaml
    variables_file = base_dir / "design/variables.yaml"
    
    try:
        with open(variables_file, 'r') as f:
            variables = yaml.safe_load(f)
        
        compute_vde(variables, args.mesh_file, args.output_dir, args.eq_file, args.use_saved_eq)
        
        # Clean up temporary OFT files after simulation
    except Exception as e:
        print(f"Error computing VDE simulation: {e}")
        import traceback
        traceback.print_exc()
        
        sys.exit(1)

if __name__ == "__main__":
    main()