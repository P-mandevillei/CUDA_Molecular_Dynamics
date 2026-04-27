import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC, abstractmethod
import math
import re
from scipy.optimize import minimize
from itertools import product
import seaborn as sns
from matplotlib import animation
import time
%pip install -U numba numba-cuda cupy-cuda12x
import numba as nb
from numba import cuda
import cupy as cp
from Forces import initialize_chain_numba, initialize_velocities_cupy
from Constants import *
from LabelKernels import calc_force_cutoff_gpu_sorted_wrapper, calc_force_cutoff_gpu_unsorted_wrapper, calc_force_cutoff_sequential_wrapper, recover_kernel
seed = 42
rng_np = np.random.default_rng(seed)
rng_cp = cp.random.default_rng(seed)
chain_lens = np.round(np.linspace(1e3, 1e4, 10)).astype(np.int32)
k = 500
r0 = 1
epsilon_attractive = 0.5
epsilon_repulsive = 1.0
sigma = 1.0

n_itr = 10

time_df = []
for chain_len in chain_lens:
    print(f"--------------- chain_len = {chain_len} ---------------")

    n_atoms = chain_len
    box_size = n_atoms * r0 * 2.5
    const_params_h = np.array([box_size, k, r0, epsilon_attractive, epsilon_repulsive, sigma])
    const_params_d = cp.array(const_params_h)
    pos = initialize_chain_numba(n_atoms, box_size, r0, rng_np, dtype = np.float64)
    v = initialize_velocities_cupy(n_atoms, target_temperature=0.5, mass=1, rng = rng_cp)
    
    positions_d = cp.array(pos)
    velocities_d = v.copy()
    forces_d = cp.zeros_like(pos)

    positions_sorted_d = cp.array(pos)
    velocities_sorted_d = v.copy()
    forces_sorted_d = cp.zeros_like(pos)
    org_idx_d = cp.arange(n_atoms, dtype=np.int32)
    
    positions_h = pos
    velocities_h = v.get()
    forces_h = np.zeros_like(pos, dtype=np.float64)

    cpu_times = []
    unsorted_times = []
    sorted_times = []
    
    for i in tqdm(range(n_itr)):
        # cpu
        cpu_start = time.perf_counter_ns()
        calc_force_cutoff_sequential_wrapper(forces_h, positions_h, const_params_h)
        cpu_end = time.perf_counter_ns()
        cpu_times.append(cpu_end - cpu_start)

        # unsorted
        start_event = cuda.event()
        end_event = cuda.event()
        start_event.record()
        calc_force_cutoff_gpu_unsorted_wrapper(forces_d, positions_d, const_params_d)
        end_event.record()
        end_event.synchronize()
        execution_time_ms = cuda.event_elapsed_time(start_event, end_event)
        unsorted_times.append(execution_time_ms * 1e6)
        if i == n_itr - 1:
            print(f"- unsorted: all close = {np.allclose(forces_h, forces_d.get())}")
        
        # sorted 
        start_event = cuda.event()
        end_event = cuda.event()
        start_event.record()
        calc_force_cutoff_gpu_sorted_wrapper(forces_sorted_d, positions_sorted_d, org_idx_d, const_params_d)
        end_event.record()
        end_event.synchronize()
        execution_time_ms = cuda.event_elapsed_time(start_event, end_event)
        sorted_times.append(execution_time_ms * 1e6)
        forces_d_recovered = cp.zeros_like(forces_sorted_d)
        n = forces_sorted_d.shape[1]
        grid_size = (n + CUTOFF_BLOCK_SIZE - 1) // CUTOFF_BLOCK_SIZE
        recover_kernel[grid_size, CUTOFF_BLOCK_SIZE](forces_d_recovered, forces_sorted_d, org_idx_d, n)
        if i == n_itr - 1:
            print(f"- sorted: all close = {np.allclose(forces_h, forces_d_recovered.get())}")
        if i == n_itr - 1:
            print(f"- sorted vs unsorted: all close = {np.allclose(np.round(forces_d.get(),0), np.round(forces_d_recovered.get(),0))}")

        
    cpu_df = pd.DataFrame({"time": cpu_times[1:], "n_atom": chain_len, "itr": range(1, n_itr), "type": "cpu"})
    unsorted_df = pd.DataFrame({"time": unsorted_times[1:], "n_atom": chain_len, "itr": range(1, n_itr), "type": "unsorted_cutoff"})
    sorted_df = pd.DataFrame({"time": sorted_times[1:], "n_atom": chain_len, "itr": range(1, n_itr), "type": "sorted_cutoff"})
    time_df.append(pd.concat([cpu_df, unsorted_df, sorted_df]))
    
time_df = pd.concat(time_df)