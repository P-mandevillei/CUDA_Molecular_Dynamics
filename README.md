# CUDA Molecular Dynamics for Linear Polymers

[Project Writeup](./deliverables/final_writeup/Project_Final_Writeup_template.pdf)

This repository contains a Molecular Dynamics (MD) simulation engine for linear polymer chains, utilizing both CPU and GPU (CUDA) implementations to accelerate force computations. The simulation employs the **Velocity Verlet** integration algorithm along with a **Velocity Rescaling Thermostat** to maintain system temperature in an NVT ensemble.

## Project Structure

The project code is divided into the following key modules:

*   **`linear_polymer_md.py`**: The main entry point for running simulations via the command-line interface.
*   **`VelocityVerlet.py`**: Implements the Velocity Verlet integration steps, temperature rescaling, and applies periodic boundary conditions. It also contains utility functions for exporting trajectories (`.traj` format) and topology files (`.psf`).
*   **`Forces.py`**: Defines the physical interactions (Harmonic bonds between adjacent monomers, and repulsive/attractive Lennard-Jones potentials). It also handles the initialization of polymer chain positions and Maxwell-Boltzmann distributed velocities.
*   **`ExactKernels.py` & `LabelKernels.py`**: These modules contain the core implementations of the force calculations.
*   **`Constants.py`**: Defines physical constants, GPU block dimensions, cutoff radii, and other global configuration values used throughout the project.

## Prerequisites

* Install the [uv](https://docs.astral.sh/uv/) package manager.

* Ensure you have a compatible NVIDIA GPU and the CUDA toolkit (version 12.x) properly installed to utilize the `cupy` and `numba.cuda` acceleration features.

## Running the Simulation

```bash
cd CUDA_Molecular_Dynamics
uv sync
uv run python linear_polymer_md.py --help
```

Fast start with 1000 atoms on the GPU:

```bash
uv run linear_polymer_md.py --n_atoms 1000 --algo implicit-matrix -v
```

### Basic Usage

The only strictly required argument is `--n_atoms` (the number of monomers in the polymer chain):

```bash
uv run linear_polymer_md.py --n_atoms 100
```

### Advanced Usage & Arguments

You can customize the simulation by supplying the following optional arguments:

#### System and Physical Parameters
*   `--n_atoms` (int, **Required**): Number of monomers in the linear polymer chain.
*   `--k` (float, default: `500.0`): Spring constant for harmonic bonds between monomers.
*   `--r0` (float, default: `1.0`): Equilibrium bond distance.
*   `--epsilon_attractive` (float, default: `0.5`): Lennard-Jones attractive well depth.
*   `--epsilon_repulsive` (float, default: `1.0`): Lennard-Jones repulsive well depth.
*   `--sigma` (float, default: `1.0`): Lennard-Jones distance parameter.
*   `--mass` (float, default: `1.0`): Mass of a single monomer.
*   `--box_size` (float, default: `n_atoms * r0 * 2.5`): Length of the cubic simulation box. Periodic boundary conditions are applied.

#### Simulation Dynamics
*   `--dt` (float, default: `0.01`): Time step for the Velocity Verlet integration.
*   `--steps` (int, default: `1001`): Total number of simulation steps to run.
*   `--temperature` (float, default: `0.5`): Target temperature for the simulation.
*   `--rescale_interval` (int, default: `100`): Frequency (in steps) at which to apply the Velocity Rescaling Thermostat.
*   `--save_interval` (int, default: `10`): Frequency (in steps) to save frames to the trajectory.
*   `--seed` (int, default: `42`): Random seed for reproducibility.

#### Output Configuration
*   `--out-top` (string): File path to save the generated system topology in `.psf` format.
*   `--out-traj` (string): File path to save the resulting simulation trajectory.
*   `-v`, `--verbose`: Enable detailed logging and progress bars.

#### Algorithm Selection
The `--algo` argument dictates how inter-particle forces are computed. 
*   **Exact Methods ($O(N^2)$ calculations):**
    *   `sequential`: Standard CPU implementation.
    *   `segmented`: Naive GPU implementation.
    *   `explicit-matrix`: GPU implementation utilizing an explicit force matrix.
    *   `implicit-matrix`: GPU implementation utilizing an implicit force matrix.
    
*   **Cutoff Methods ($O(N)$ calculations):**
    *   `cutoff-sequential`: CPU implementation utilizing cutoff distances.
    *   `cutoff-unsorted`: GPU implementation utilizing cutoffs.
    *   `cutoff-sorted`: Optimized GPU cutoff implementation that reduces the cost of re-sorting labels.
