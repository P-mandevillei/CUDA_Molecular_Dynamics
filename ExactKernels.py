import numpy as np
import math
from scipy.optimize import minimize
import numba as nb
from numba import cuda
import cupy as cp
import nvtx

from Constants import *
from Forces import *

# -------------- Compare: Sequential ---------------------
@nb.njit
def calc_force_sequential(
  forces: np.ndarray, # global forces array (output)
  positions: np.ndarray, # global positions array (input)
  const_params: np.ndarray
):
  forces.fill(0)
  params = const_params
  n_particles = positions.shape[1]
  for i in range(n_particles-1):
    for j in range(i+1, n_particles):
      distance = 0
      for k in range(SPACE_N_DIM):
        dx = minimum_image_1d(positions[k, i] - positions[k, j], params[BOX_SIZE_IDX])
        distance += dx**2
      distance = math.sqrt(distance)
      magnitude = calc_force(distance, params, j - i)
      for k in range(SPACE_N_DIM):
        f = magnitude * minimum_image_1d(positions[k, i] - positions[k, j], params[BOX_SIZE_IDX]) / max(distance, DIV_BY_ZERO_GUARD)
        forces[k, i] += f
        forces[k, j] -= f

# ------------ Method 0: Segmented Traversal ------------------
@cuda.jit
def calc_force_segmented(
  forces: cp.ndarray, # global forces array (output)
  positions: cp.ndarray, # global positions array (input)
  const_params: cp.ndarray
):
  dtype = nb.float64
  params = cuda.const.array_like(const_params)

  idx = cuda.grid(1)
  receiver_position = cuda.local.array(shape = SPACE_N_DIM, dtype = dtype)
  if idx < positions.shape[1]:
    for dim in range(SPACE_N_DIM):
      receiver_position[dim] = positions[dim, idx]

  src_positions = cuda.shared.array(shape = (SEGMENT, 3), dtype = dtype)
  force = cuda.local.array(shape = SPACE_N_DIM, dtype = dtype)
  for i in range(SPACE_N_DIM):
    force[i] = 0

  for start in range(0, positions.shape[1], SEGMENT):
    sweep = min(SEGMENT, positions.shape[1] - start)
    # cooperate to load src positions
    if cuda.threadIdx.x < sweep:
      for dim in range(SPACE_N_DIM):
        src_positions[cuda.threadIdx.x, dim] = positions[dim, start + cuda.threadIdx.x]
    cuda.syncthreads()

    # calculate
    if idx < positions.shape[1]:
      for i in range(sweep):
        distance = 0
        for dim in range(SPACE_N_DIM):
          distance += minimum_image_1d(receiver_position[dim] - src_positions[i, dim], params[BOX_SIZE_IDX])**2
        distance = math.sqrt(distance)
        magnitude = calc_force(
          distance,
          params,
          idx - (start + i)
        )
        for dim in range(SPACE_N_DIM):
          force[dim] += magnitude * minimum_image_1d(receiver_position[dim] - src_positions[i, dim], params[BOX_SIZE_IDX]) / max(distance, DIV_BY_ZERO_GUARD)
    cuda.syncthreads()
  if idx < positions.shape[1]:
    for dim in range(SPACE_N_DIM):
      forces[dim, idx] = force[dim] 

# ------------ Method 1: Force Matrix ------------------
@cuda.jit
def calc_force_matrix(
  forces: cp.ndarray, # global forces array (output)
  positions: cp.ndarray, # global positions array (input)
  tile_n: int,
  const_params: cp.ndarray
):
  dtype = nb.float64
  params = cuda.const.array_like(const_params)

  # heavy index calculation once per block
  
  block_idx = cuda.shared.array(shape = 2, dtype = nb.int32) # row, col
  if cuda.threadIdx.x == 0:
    row = math.ceil(tile_n - 0.5 - math.sqrt((2*tile_n+1)*(2*tile_n+1) - 8*(cuda.blockIdx.x+1))/2)
    col = cuda.blockIdx.x + row - (2*tile_n-row+1)*row//2
    block_idx[0] = row * FORCE_MAT_TILE_DIM
    block_idx[1] = col * FORCE_MAT_TILE_DIM
  cuda.syncthreads()

  # load source positions to shared memory
  sweep = min(FORCE_MAT_TILE_DIM, forces.shape[1]-block_idx[0])
  src_positions = cuda.shared.array(shape = (FORCE_MAT_TILE_DIM, 3), dtype = dtype)
  if cuda.threadIdx.x < sweep:
    for dim in range(SPACE_N_DIM):
      src_positions[cuda.threadIdx.x, dim] = positions[dim, block_idx[0]+cuda.threadIdx.x]
  cuda.syncthreads()

  receiver_idx = block_idx[1] + cuda.threadIdx.x

  # ---------------------- diagonal --------------------------------
  if block_idx[0] == block_idx[1]:
    if receiver_idx < forces.shape[1]: # this guarantees threadIdx < sweep since the force matrix is a square matrix
      # load receiver positions
      receiver_position = cuda.local.array(shape = 3, dtype = dtype)
      for dim in range(SPACE_N_DIM):
        receiver_position[dim] = src_positions[cuda.threadIdx.x, dim]
      
      col_sum = cuda.local.array(shape = SPACE_N_DIM, dtype = dtype)
      for i in range(SPACE_N_DIM):
        col_sum[i] = 0
      for i in range(1, sweep): # start from 1 to skip calculating forces by itself, avoid division by 0 distance
        # juxtapose src indices
        # when not diagonal, minimize shared memory contention
        src_idx = (i + cuda.threadIdx.x) % sweep
        distance = 0
        for dim in range(SPACE_N_DIM):
          distance += minimum_image_1d(receiver_position[dim] - src_positions[src_idx, dim], params[BOX_SIZE_IDX])**2
        distance = math.sqrt(distance)
        magnitude = calc_force(
          distance,
          params,
          receiver_idx - (block_idx[0] + src_idx)
        )
        for dim in range(SPACE_N_DIM):
          col_sum[dim] += magnitude * minimum_image_1d(receiver_position[dim] - src_positions[src_idx, dim], params[BOX_SIZE_IDX]) / max(distance, DIV_BY_ZERO_GUARD)
      for dim in range(SPACE_N_DIM):
        cuda.atomic.add(forces, (dim, receiver_idx), col_sum[dim])
  
  # ---------------------- non-diagonal --------------------------------
  else:
    row_sums = cuda.shared.array(shape = (FORCE_MAT_TILE_DIM, 3), dtype = dtype) # sweep must be equal to tile dim when it's not a diagonal block
    for i in range(SPACE_N_DIM):
      row_sums[cuda.threadIdx.x, i] = 0
    cuda.syncthreads()

    if receiver_idx < forces.shape[1]:
      col_sum = cuda.local.array(shape = SPACE_N_DIM, dtype = dtype)
      for i in range(SPACE_N_DIM):
        col_sum[i] = 0
      receiver_position = cuda.local.array(shape = SPACE_N_DIM, dtype = dtype)
      for dim in range(SPACE_N_DIM):
        receiver_position[dim] = positions[dim, receiver_idx]
      
      for i in range(sweep):
        src_idx = (i + cuda.threadIdx.x) % sweep
        distance = 0
        for dim in range(SPACE_N_DIM):
          distance += minimum_image_1d(receiver_position[dim] - src_positions[src_idx, dim], params[BOX_SIZE_IDX])**2
        distance = math.sqrt(distance)
        magnitude = calc_force(
          distance,
          params,
          receiver_idx - (block_idx[0] + src_idx)
        )
        for dim in range(SPACE_N_DIM):
          f = magnitude * minimum_image_1d(receiver_position[dim] - src_positions[src_idx, dim], params[BOX_SIZE_IDX]) / max(distance, DIV_BY_ZERO_GUARD)
          col_sum[dim] += f
          cuda.atomic.add(row_sums, (src_idx, dim), -f) # no race condition when FORCE_MAT_TILE_DIM <= WARP_SIZE
      for dim in range(SPACE_N_DIM):
        cuda.atomic.add(forces, (dim, receiver_idx), col_sum[dim])

    cuda.syncthreads()

    if cuda.threadIdx.x < sweep:
      for dim in range(SPACE_N_DIM):
        cuda.atomic.add(forces, (dim, cuda.threadIdx.x+block_idx[0]), row_sums[cuda.threadIdx.x, dim])

# ---------- Method 2: Explicit Force Matrix ----------
@cuda.jit
def calc_force_matrix_explicit(
  forces_mat: cp.ndarray, # global forces matrix array (1d) (output)
  positions: cp.ndarray, # global positions array (input)
  const_params: cp.ndarray
):
  dtype = nb.float64
  params = cuda.const.array_like(const_params)
  idx = cuda.grid(1)
  n = positions.shape[1] - 1 # -1 since we don't need the diagonal entries
  if idx >= n*(n+1)//2:
    return

  row = int(math.ceil(n - 0.5 - math.sqrt((2*n+1)*(2*n+1) - 8*(idx+1))/2))
  col = idx + row - (2*n-row+1)*row//2 + 1 # +1 since we don't need the diagonal entries
  
  src_positions = cuda.local.array(SPACE_N_DIM, dtype = dtype)
  receiver_positions = cuda.local.array(SPACE_N_DIM, dtype = dtype)
  for dim in range(SPACE_N_DIM):
    src_positions[dim] = positions[dim, row]
    receiver_positions[dim] = positions[dim, col]
  distance = 0
  for dim in range(SPACE_N_DIM):
    distance += minimum_image_1d(receiver_positions[dim] - src_positions[dim], params[BOX_SIZE_IDX])**2
  distance = math.sqrt(distance)
  magnitude = calc_force(
    distance,
    params,
    col - row
  )
  for dim in range(SPACE_N_DIM):
    forces_mat[dim, idx] = magnitude * minimum_image_1d(receiver_positions[dim] - src_positions[dim], params[BOX_SIZE_IDX]) / max(distance, DIV_BY_ZERO_GUARD)

@cuda.jit
def force_mat_sum(
  forces_mat: cp.ndarray, # upper diagonal matrix stored as 1d array
  forces: cp.ndarray # output array
):
  dtype = nb.float64
  coarsen_factor = (forces.shape[1] + FORCE_MAT_REDUC_BLOCK_DIM - 1) // FORCE_MAT_REDUC_BLOCK_DIM
  n = forces.shape[1] - 1
  results = cuda.shared.array(shape = (FORCE_MAT_REDUC_BLOCK_DIM, SPACE_N_DIM), dtype = dtype)

  result = cuda.local.array(shape = SPACE_N_DIM, dtype = dtype)
  for dim in range(SPACE_N_DIM):
    result[dim] = 0
  for idx in range(cuda.threadIdx.x*coarsen_factor, min(forces.shape[1], (cuda.threadIdx.x+1)*coarsen_factor)):
    col = cuda.blockIdx.x
    row = idx
    if row == col:
      continue
    coeff = 1
    if row > col:
      row, col = col, row
      coeff = -1
    col -= 1
    mat_idx = (2*n-row+1)*row//2 + col - row
    for dim in range(SPACE_N_DIM):
      result[dim] += coeff*forces_mat[dim, mat_idx]
  for dim in range(SPACE_N_DIM):
    results[cuda.threadIdx.x, dim] = result[dim]
  cuda.syncthreads()

  block = FORCE_MAT_REDUC_BLOCK_DIM // 2
  while block > 0:
    if cuda.threadIdx.x < block:
      for dim in range(SPACE_N_DIM):
        results[cuda.threadIdx.x, dim] += results[cuda.threadIdx.x + block, dim]
    block //= 2
    cuda.syncthreads()
  
  if cuda.threadIdx.x == 0:
    for dim in range(SPACE_N_DIM):
      forces[dim, cuda.blockIdx.x] = results[0, dim]

# -------------------------------------------- API --------------------------------------------
def calc_force_matrix_wrapper(forces, positions, const_params):
  with nvtx.annotate("Implicit Matrix", color="blue"):
    forces.fill(0)
    tile_n = math.ceil(positions.shape[1] / FORCE_MAT_TILE_DIM)
    grid_dim = tile_n * (1 + tile_n) // 2
    calc_force_matrix[grid_dim, FORCE_MAT_TILE_DIM](forces, positions, tile_n, const_params)

def calc_force_segmented_wrapper(forces, positions, const_params):
  with nvtx.annotate("Segmented Traversal", color="red"):
    grid_dim = (positions.shape[1] + SEGMENT - 1) // SEGMENT
    calc_force_segmented[grid_dim, SEGMENT](forces, positions, const_params)

def calc_force_matrix_explicit_wrapper(forces, positions, const_params):
  with nvtx.annotate("Explicit Matrix", color="green"):
    chain_len = positions.shape[1]
    n_threads = (chain_len-1) * (1 + (chain_len-1)) // 2
    grid_dim = (n_threads + FORCE_MAT_EXPLICIT_BLOCK_DIM - 1) // FORCE_MAT_EXPLICIT_BLOCK_DIM
    forces_mat_d = cp.zeros(shape = (SPACE_N_DIM, n_threads), dtype = cp.float64)
  calc_force_matrix_explicit[grid_dim, FORCE_MAT_EXPLICIT_BLOCK_DIM](forces_mat_d, positions, const_params)
  force_mat_sum[chain_len, FORCE_MAT_REDUC_BLOCK_DIM](forces_mat_d, forces)