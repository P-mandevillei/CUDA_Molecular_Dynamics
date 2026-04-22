from Forces import initialize_chain_numba, initialize_velocities_cupy, initialize_velocities_cpu
from VelocityVerlet import write_traj, run_md, write_psf
from ExactKernels import calc_force_sequential, calc_force_matrix_wrapper, calc_force_segmented_wrapper, calc_force_matrix_explicit_wrapper
import argparse
import numpy as np

ALGO_CHOICE = ['sequential', 'segmented', 'implicit-matrix', 'explicit-matrix']

def parse_simulation_args():
    parser = argparse.ArgumentParser(
        description="Molecular Dynamics on a Linear Polymer with Velocity Verlet Integration and Velocity Rescaling Thermostat."
    )
    # Required argument
    parser.add_argument("--n_atoms", type=int, required=True,
                        help="Number of monomers in the simulation (Required)")

    # Optional arguments with static defaults
    parser.add_argument("--k", type=float, default=500.0, help="Spring constant (default: 500.0 L.J.U.)")
    parser.add_argument("--r0", type=float, default=1.0, help="Equilibrium distance (default: 1.0 L.J.U.)")
    parser.add_argument("--epsilon_attractive", type=float, default=0.5, help="Attractive epsilon (default: 0.5 L.J.U.)")
    parser.add_argument("--epsilon_repulsive", type=float, default=1.0, help="Repulsive epsilon (default: 1.0 L.J.U.)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma parameter (default: 1.0 L.J.U.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step (default: 0.01 L.J.U.)")
    parser.add_argument("--mass", type=float, default=1.0, help="Particle mass (default: 1.0 L.J.U.)")
    parser.add_argument("--temperature", type=float, default=0.5, help="System temperature (default: 0.5 L.J.U.)")
    parser.add_argument("--steps", type=int, default=1001, help="Number of simulation steps (default: 1001)")
    parser.add_argument("--rescale_interval", type=int, default=100, help="Steps between temperature rescaling (default: 100)")
    parser.add_argument("--save_interval", type=int, default=10, help="Steps between saving frames (default: 10)")
    parser.add_argument(
        '--algo', 
        choices=ALGO_CHOICE, 
        default='sequential',
        help="Defines the force computation algorithm."
    )
    parser.add_argument("--out-top", type=str, default=None, help="Output topology file path")
    parser.add_argument("--out-traj", type=str, default=None, help="Output trajectory file path")
    
    # Optional argument with a dynamic default
    parser.add_argument("--box_size", type=float, default=None,
                        help="Size of the simulation box (default: n_atoms * r0 * 2.5)")
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    if args.box_size is None:
        args.box_size = args.n_atoms * args.r0 * 2.5

    return args

if __name__ == "__main__":
    args = parse_simulation_args()

    n_atoms = args.n_atoms
    k = args.k
    r0 = args.r0
    box_size = args.box_size
    epsilon_attractive = args.epsilon_attractive
    epsilon_repulsive = args.epsilon_repulsive
    sigma = args.sigma
    seed = args.seed
    dt = args.dt
    mass = args.mass
    temperature = args.temperature
    steps = args.steps
    rescale_interval = args.rescale_interval
    save_interval = args.save_interval
    verbose = args.verbose
    algo = args.algo
    traj = args.out_traj
    top = args.out_top

    match algo:
        case 'sequential':
            calc_force_func = calc_force_sequential
            device = False
        case 'segmented':
            calc_force_func = calc_force_segmented_wrapper
            device = True
        case 'implicit-matrix':
            calc_force_func = calc_force_matrix_wrapper
            device = True
        case 'explicit-matrix':
            calc_force_func = calc_force_matrix_explicit_wrapper
            device = True

    if device:
        import cupy as cp

    rng_np = np.random.default_rng(seed)
    if device:
        rng_cp = cp.random.default_rng(seed)
        const_params = cp.array([box_size, k, r0, epsilon_attractive, epsilon_repulsive, sigma])
        pos = cp.array(initialize_chain_numba(n_atoms, box_size, r0, rng_np, dtype = np.float64))
        v = initialize_velocities_cupy(n_atoms, target_temperature=temperature, mass=mass, rng = rng_cp)
        forces = cp.zeros_like(pos)
    else:
        const_params = np.array([box_size, k, r0, epsilon_attractive, epsilon_repulsive, sigma])
        pos = initialize_chain_numba(n_atoms, box_size, r0, rng_np, dtype = np.float64)
        v = initialize_velocities_cpu(n_atoms, target_temperature=temperature, mass=mass, rng = rng_np)

    if args.verbose:
        print("--- Simulation Parameters ---")
        print(f"n_atoms:             {n_atoms}")
        print(f"algo:                {algo}")
        print(f"k:                   {k}")
        print(f"r0:                  {r0}")
        print(f"box_size:            {box_size}")
        print(f"epsilon_attractive:  {epsilon_attractive}")
        print(f"epsilon_repulsive:   {epsilon_repulsive}")
        print(f"sigma:               {sigma}")
        print(f"seed:                {seed}")
        print(f"dt:                  {dt}")
        print(f"mass:                {mass}")
        print(f"temperature:         {temperature}")
        print(f"steps:               {steps}")
        print(f"rescale_interval:    {rescale_interval}")
        print(f"save_interval:       {save_interval}")
        print("-----------------------------")
    
    frames = run_md(
        pos,
        v,
        const_params,
        calc_force_func,
        dt,
        mass,
        temperature,
        steps, box_size,
        rescale_interval,
        save_interval,
        device = device,
        verbose = verbose
    )

    if top:
        write_psf(top, n_atoms, mass)
        if verbose:
            print(f"Saved topology to {top}")
    if traj:
        write_traj(traj, frames)
        if verbose:
            print(f"Saved trajectory to {traj}")