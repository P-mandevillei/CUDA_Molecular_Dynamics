import numpy as np
import math
from scipy.optimize import minimize
import numba as nb
from numba import cuda
import cupy as cp

from Constants import *

# -------------------------------- Initialize ----------------------------------------
@nb.njit
def assign_position(positions, new_positions, idx):
    positions[0, idx] = new_positions[0]
    positions[1, idx] = new_positions[1]
    positions[2, idx] = new_positions[2]

@nb.njit
def initialize_chain_numba(
    n_particles: int,
    box_size: float,
    r0: float,
    rng: np.random._generator.Generator,
    dtype=np.float32
) -> np.ndarray:
    """
    Randomly initialize atom positions by growing the chain
    """
    positions = np.zeros((3, n_particles), dtype=dtype)

    current_position = np.array([box_size / 2, box_size / 2, box_size / 2], dtype=dtype)
    assign_position(positions, current_position, 0)

    for i in range(1, n_particles):
        direction = rng.normal(size = 3).astype(dtype)
        norm = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
        direction /= norm

        next_position = current_position + r0 * direction
        next_position = (next_position % box_size).astype(dtype) # pbc
        assign_position(positions, next_position, i)
        current_position = next_position

    return positions

def initialize_velocities_cpu(
    n_particles: int,
    target_temperature: float, 
    mass: float,
    rng: np.random._generator.Generator,
    kB=1.0,
    dtype=np.float64
) -> np.ndarray:
    """
    Initialize particle velocities by drawing from the Maxwell-Botzmann distribution at target temperature
    """
    sigma = np.sqrt(kB * target_temperature / mass).astype(dtype)

    velocities = rng.normal(
        size=(3, n_particles)
    ).astype(dtype)
    velocities = velocities * sigma

    # Remove center-of-mass velocity
    velocities -= np.mean(velocities, axis=-1).reshape(3, 1)

    return velocities

def initialize_velocities_cupy(
    n_particles: int,
    target_temperature: float, 
    mass: float,
    rng: cp.random._generator_api.Generator,
    kB=1.0,
    dtype=cp.float64
) -> cp.ndarray:
    """
    Initialize particle velocities by drawing from the Maxwell-Botzmann distribution at target temperature
    """
    sigma = cp.sqrt(kB * target_temperature / mass).astype(dtype)

    velocities = rng.standard_normal(
        size=(3, n_particles), 
        dtype=dtype
    )
    velocities = velocities * sigma

    # Remove center-of-mass velocity
    velocities -= cp.mean(velocities, axis=-1).reshape(3, 1)

    return velocities

# -------------------------------- Define Forces --------------------------------
@nb.njit
def harmonic_force(r, k, r0):
  return -k * (r-r0)

well_coeff = 2**(1/6)
@nb.njit
def repulsive_lj_force(r, sigma, epsilon):
  if r < well_coeff*sigma:
    r = max(r, DIV_BY_ZERO_GUARD)
    sixth_pow = (sigma/r) ** 6
    return -24*epsilon/r * (sixth_pow - 2*(sixth_pow**2))
  else:
    return 0

@nb.njit
def attractive_lj_force(r, sigma, epsilon):
  r = max(r, DIV_BY_ZERO_GUARD)
  sixth_pow = (sigma/r) ** 6
  return -24*epsilon/r * (sixth_pow - 2*(sixth_pow**2))

@nb.njit
def calc_force(distance, params, type):
  if type == HARMONIC or type == -HARMONIC:
     return harmonic_force(distance, params[K_IDX], params[R0_IDX])
  if type == REPULSIVE or type == -REPULSIVE:
     return repulsive_lj_force(distance, params[SIGMA_IDX], params[EPSILON_REPULSIVE_IDX])
  if type >= ATTRACTIVE or type <= -ATTRACTIVE:
     return attractive_lj_force(distance, params[SIGMA_IDX], params[EPSILON_ATTRACTIVE_IDX])
  return 0

@nb.njit
def minimum_image_1d(dx, box_size):
    return dx - box_size * math.floor((dx / box_size) + 0.5)
