import math
from math import floor
import numpy as np
import numba as nb
from numba import cuda
import cupy as cp
from Constants import *
from Forces import calc_force, minimum_image_1d

@cuda.jit
def atom_label_kernel(
    labels, # global labels array (output)
    positions, # global positions array (input)
    cell_size: float, # size of each cell
    dim: int # number of cell in each dimension
):
    idx = cuda.grid(1)
    if idx >= positions.shape[1]:
        return
    x = positions[0, idx]
    y = positions[1, idx]
    z = positions[2, idx]
    cell_x = int(math.floor(x / cell_size)) % dim
    cell_y = int(math.floor(y / cell_size)) % dim
    cell_z = int(math.floor(z / cell_size)) % dim
    
    labels[idx] = cell_x + cell_y * dim + cell_z * dim * dim

# positions, forces are unsorted, so that forces[i] corresponds to positions[i]
@cuda.jit
def calc_force_cutoff_gpu_unsorted(
    forces,
    cell_start,
    positions,
    org_idx,
    cell_size,
    box_size,
    n_cells,
    cutoff,
    const_params
):
    i = cuda.grid(1)

    if i >= positions.shape[1]:
        return

    i_o = org_idx[i]

    xi = positions[0, i]
    yi = positions[1, i]
    zi = positions[2, i]

    xw = xi
    yw = yi
    zw = zi

    while xw >= box_size:
        xw -= box_size
    while xw < 0:
        xw += box_size

    while yw >= box_size:
        yw -= box_size
    while yw < 0:
        yw += box_size

    while zw >= box_size:
        zw -= box_size
    while zw < 0:
        zw += box_size

    cx = int(xw / cell_size)
    cy = int(yw / cell_size)
    cz = int(zw / cell_size)

    force_x = 0.0
    force_y = 0.0
    force_z = 0.0

    for dz in range(-1, 2):
        nz = cz + dz
        if nz < 0:
            nz += n_cells
        elif nz >= n_cells:
            nz -= n_cells

        for dy in range(-1, 2):
            ny = cy + dy
            if ny < 0:
                ny += n_cells
            elif ny >= n_cells:
                ny -= n_cells

            for dx in range(-1, 2):
                nx = cx + dx
                if nx < 0:
                    nx += n_cells
                elif nx >= n_cells:
                    nx -= n_cells

                nbr = nx + ny * n_cells + nz * n_cells * n_cells

                start = cell_start[nbr]
                end = cell_start[nbr + 1]

                for j in range(start, end):
                    if j == i:
                        continue

                    xj = positions[0, j]
                    yj = positions[1, j]
                    zj = positions[2, j]

                    xij = minimum_image_1d(xi - xj, box_size)
                    yij = minimum_image_1d(yi - yj, box_size)
                    zij = minimum_image_1d(zi - zj, box_size)

                    r2 = xij * xij + yij * yij + zij * zij
                    if r2 < DIV_BY_ZERO_GUARD * DIV_BY_ZERO_GUARD:
                        continue

                    rij = math.sqrt(r2)

                    if rij > cutoff:
                        continue

                    j_o = org_idx[j]
                    sep = j_o - i_o

                    fmag = calc_force(rij, const_params, sep)

                    force_x += fmag * xij / rij
                    force_y += fmag * yij / rij
                    force_z += fmag * zij / rij

    forces[0, i_o] = force_x
    forces[1, i_o] = force_y
    forces[2, i_o] = force_z


# positions, forces are sorted by original indices, so that forces[i_o] corresponds to positions[i_o]
@cuda.jit
def calc_force_cutoff_gpu_sorted(
    forces,
    cell_start,
    positions,
    org_idx,
    cell_size,
    box_size,
    n_cells,
    cutoff,
    const_params
):
    params = cuda.const.array_like(const_params)
    i = cuda.grid(1)

    if i >= positions.shape[1]:
        return

    i_o = org_idx[i]

    xi = positions[0, i]
    yi = positions[1, i]
    zi = positions[2, i]

    xw = xi
    yw = yi
    zw = zi

    while xw >= box_size:
        xw -= box_size
    while xw < 0:
        xw += box_size

    while yw >= box_size:
        yw -= box_size
    while yw < 0:
        yw += box_size

    while zw >= box_size:
        zw -= box_size
    while zw < 0:
        zw += box_size

    cx = int(xw / cell_size)
    cy = int(yw / cell_size)
    cz = int(zw / cell_size)

    force_x = 0.0
    force_y = 0.0
    force_z = 0.0

    for dz in range(-1, 2):
        nz = cz + dz
        if nz < 0:
            nz += n_cells
        elif nz >= n_cells:
            nz -= n_cells

        for dy in range(-1, 2):
            ny = cy + dy
            if ny < 0:
                ny += n_cells
            elif ny >= n_cells:
                ny -= n_cells

            for dx in range(-1, 2):
                nx = cx + dx
                if nx < 0:
                    nx += n_cells
                elif nx >= n_cells:
                    nx -= n_cells

                nbr = nx + ny * n_cells + nz * n_cells * n_cells

                start = cell_start[nbr]
                end = cell_start[nbr + 1]

                for j in range(start, end):
                    if j == i:
                        continue

                    xj = positions[0, j]
                    yj = positions[1, j]
                    zj = positions[2, j]

                    xij = minimum_image_1d(xi - xj, params[BOX_SIZE_IDX])
                    yij = minimum_image_1d(yi - yj, params[BOX_SIZE_IDX])
                    zij = minimum_image_1d(zi - zj, params[BOX_SIZE_IDX])

                    r2 = xij * xij + yij * yij + zij * zij
                    if r2 < DIV_BY_ZERO_GUARD * DIV_BY_ZERO_GUARD:
                        continue

                    rij = math.sqrt(r2)

                    if rij > cutoff:
                        continue

                    j_o = org_idx[j]
                    sep = abs(j_o - i_o)

                    fmag = calc_force(rij, params, sep)

                    force_x += fmag * xij / rij
                    force_y += fmag * yij / rij
                    force_z += fmag * zij / rij

    forces[0, i] = force_x
    forces[1, i] = force_y
    forces[2, i] = force_z


def calc_force_cutoff_gpu_sorted_wrapper(forces, positions, org_idx, const_params):
    labels_d = cp.zeros(positions.shape[1], dtype=cp.int32)
    dim = CELL_DIM
    cell_size = const_params[BOX_SIZE_IDX] / dim
    box_size = const_params[BOX_SIZE_IDX]
    cutoff = const_params[SIGMA_IDX] * CUTOFF_COEFF
    n_total_cells = dim ** 3
    grid_size = (positions.shape[1] + CUTOFF_BLOCK_SIZE - 1) // CUTOFF_BLOCK_SIZE

    # label stage
    atom_label_kernel[grid_size, CUTOFF_BLOCK_SIZE](labels_d, positions, cell_size, dim)
    cuda.synchronize()

    # sort + build cell_start stage
    order = cp.argsort(labels_d)
    labels_sorted = labels_d[order]
    prev_org_idx = org_idx.copy()
    positions[:] = positions[:, order]
    org_idx[:] = prev_org_idx[order]


    counts = cp.bincount(labels_sorted, minlength=n_total_cells)
    cell_start = cp.zeros(n_total_cells + 1, dtype=cp.int32)
    cell_start[1:] = cp.cumsum(counts)

    # force stage
    cell_start_nb = cuda.as_cuda_array(cell_start)
    positions_nb = cuda.as_cuda_array(positions)
    orig_idx_nb = cuda.as_cuda_array(org_idx)
    calc_force_cutoff_gpu_sorted[grid_size, CUTOFF_BLOCK_SIZE](
        forces,
        cell_start_nb,
        positions_nb,
        orig_idx_nb,
        cell_size,
        box_size,
        dim,
        cutoff,
        const_params
    )
    cuda.synchronize()
    return order


def calc_force_cutoff_gpu_unsorted_wrapper(forces, positions, const_params):
    labels_d = cp.zeros(positions.shape[1], dtype=cp.int32)
    dim = CELL_DIM
    cell_size = const_params[BOX_SIZE_IDX] / dim
    box_size = const_params[BOX_SIZE_IDX]
    cutoff = const_params[SIGMA_IDX] * CUTOFF_COEFF
    n_total_cells = dim ** 3
    grid_size = (positions.shape[1] + CUTOFF_BLOCK_SIZE - 1) // CUTOFF_BLOCK_SIZE

    # label stage
    atom_label_kernel[grid_size, CUTOFF_BLOCK_SIZE](labels_d, positions, cell_size, dim)
    cuda.synchronize()

    # sort + build cell_start stage
    order = cp.argsort(labels_d)
    labels_sorted = labels_d[order]
    positions_sorted = cp.empty_like(positions)
    positions_sorted[:] = positions[:, order]
    org_idx = cp.arange(positions.shape[1], dtype=cp.int32)
    org_idx[:] = order

    counts = cp.bincount(labels_sorted, minlength=n_total_cells)
    cell_start = cp.zeros(n_total_cells + 1, dtype=cp.int32)
    cell_start[1:] = cp.cumsum(counts)

    # force stage
    cell_start_nb = cuda.as_cuda_array(cell_start)
    positions_sorted_nb = cuda.as_cuda_array(positions_sorted)
    orig_idx_nb = cuda.as_cuda_array(org_idx)
    calc_force_cutoff_gpu_unsorted[grid_size, CUTOFF_BLOCK_SIZE](
        forces,
        cell_start_nb,
        positions_sorted_nb,
        orig_idx_nb,
        cell_size,
        box_size,
        dim,
        cutoff,
        const_params
    )
    cuda.synchronize()

@cuda.jit
def recover_kernel(recover_array, candidate_array, org_idx, n):
    idx = cuda.grid(1)
    if idx >= n:
        return
    original_position = org_idx[idx]
    recover_array[0, original_position] = candidate_array[0, idx]
    recover_array[1, original_position] = candidate_array[1, idx]
    recover_array[2, original_position] = candidate_array[2, idx]


# ---------------------------- Sequential Cutoff Kernels --------------------------------
@nb.njit
def calc_force_cutoff_sequential(
  forces: np.ndarray, # global forces array (output)
  positions: np.ndarray, # global positions array (input)
  cutoff: float,
  const_params: np.ndarray
):
  forces.fill(0)
  params = cuda.const.array_like(const_params)
  n_particles = positions.shape[1]
  for i in range(n_particles-1):
    for j in range(i+1, n_particles):
      distance = 0
      for k in range(SPACE_N_DIM):
        dx = minimum_image_1d(positions[k, i] - positions[k, j], params[BOX_SIZE_IDX])
        distance += dx**2
      distance = math.sqrt(distance)
      if distance > cutoff:
        continue
      magnitude = calc_force(distance, params, j - i)
      for k in range(SPACE_N_DIM):
        f = magnitude * minimum_image_1d(positions[k, i] - positions[k, j], params[BOX_SIZE_IDX]) / max(distance, DIV_BY_ZERO_GUARD)
        forces[k, i] += f
        forces[k, j] -= f

def calc_force_cutoff_sequential_wrapper(forces, positions, const_params):
    calc_force_cutoff_sequential(forces, positions, const_params[SIGMA_IDX] * CUTOFF_COEFF)