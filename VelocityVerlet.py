import numpy as np
import math
from scipy.optimize import minimize
import numba as nb
from numba import cuda
import cupy as cp
from tqdm.auto import tqdm

from Constants import *

def write_psf(filename, n, mass):
    """
    Writes a .psf file to the given file path.
    """
    with open(filename, 'w') as file:
        file.write("PSF\n\n")
        file.write("       1 !NTITLE\n")
        file.write(" REMARKS coarse-grained polymer chain\n\n")

        file.write(f"{n:8d} !NATOM\n")
        for i in range(1, n+1):
            # index segid resid resname atomname atomtype charge mass
            file.write(
                f"{i:8d} POLY   1 POL  CG    CG    0.000  {mass:3.3f}\n"
            )
        file.write("\n")

        file.write(f"{n-1:8d} !NBOND: bonds\n")
        n_bonds = 0
        for i in range(n-1):
            file.write(f"{i+1:8d}{i+2:8d}")
            n_bonds += 1
            if (n_bonds % 4 == 0):
                file.write("\n")
        file.write("\n")

def rescale_d(velocities: cp.ndarray, mass: float, temperature: float):
  n = velocities.shape[1]
  ke = 1/2*cp.sum(mass * (cp.linalg.norm(velocities, axis=0)**2))
  cur_temp = 2/3 * ke / n
  velocities *= cp.sqrt(temperature / cur_temp)

def rescale_h(velocities: np.ndarray, mass: float, temperature: float):
  n = velocities.shape[1]
  ke = 1/2*np.sum(mass * (np.linalg.norm(velocities, axis=0)**2))
  cur_temp = 2/3 * ke / n
  velocities *= np.sqrt(temperature / cur_temp)

def write_traj(filename, frames):
  n = frames[0].shape[1]
  with open(filename, "w") as f:
    for pos in frames:
      f.write(f"{n}\nframe\n")
      for x, y, z in zip(pos[0], pos[1], pos[2]):
        if x<0 or y<0 or z<0:
          print(f"({x}, {y}, {z})")
        f.write(f"CG {x:.6f} {y:.6f} {z:.6f}\n")

# host function
def run_md(
  positions: cp.ndarray | np.ndarray, # 3 * n
  velocities: cp.ndarray | np.ndarray, # 3 * n
  const_params: cp.ndarray | np.ndarray,
  compute_forces_func, # must have signature compute_forces_func(forces, positions, const_params)
  dt: float,
  mass: float,
  temperature: float,
  steps: int,
  box_size: float,
  rescale_interval: int,
  save_interval: int,
  device = False,
  verbose = True
):
  frames = np.zeros(shape = ((steps-1)//save_interval+1, 3, positions.shape[1]), dtype = np.float64)

  if device:
    forces = cp.zeros(shape = (3, positions.shape[1]), dtype = cp.float64)
  else:
    forces = np.zeros(shape = (3, positions.shape[1]), dtype = np.float64)

  compute_forces_func(forces, positions, const_params)
  velocities += 0.5 * forces / mass * dt
  positions += velocities * dt
  if device:
    positions -= box_size * cp.floor(positions / box_size)
  else:
    positions -= box_size * np.floor(positions / box_size)
  if device:
    frames[0] = positions.get()
  else:
    frames[0] = positions.copy()

  traversal = tqdm(range(1, steps)) if verbose else range(1, steps)
  for step in traversal:
    compute_forces_func(forces, positions, const_params)
    dv = forces / mass * dt

    if (step-1) % rescale_interval == 0:
      velocities += 0.5 * dv
      if device:
        rescale_d(velocities, mass, temperature)
      else:
        rescale_h(velocities, mass, temperature)
      velocities += 0.5 * dv
    else:
      velocities += dv

    positions += velocities * dt
    if device:
      positions -= box_size * cp.floor(positions / box_size)
    else:
      positions -= box_size * np.floor(positions / box_size)
    if step % save_interval == 0:
      if device:
        frames[step // save_interval] = positions.get()
      else:
        frames[step // save_interval] = positions.copy()
  
  return frames