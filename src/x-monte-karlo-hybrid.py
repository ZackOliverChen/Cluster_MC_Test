import os
import sys
import time
import numpy as np
import cupy as cp
from dask.distributed import Client
from numba import njit, prange, uint64

# Run this script from shell using the following command:
# LD_LIBRARY_PATH=/home/zack/Programming/.dask-venv/lib /home/zack/Programming/.dask-venv/bin/python /home/zack/Programming/FirstTestProject/src/monte-karlo-hybrid.py
# ---------------------------------------------------------------------------
# HARDWARE OVERRIDES: Direct-inject your virtual environment path maps
# ---------------------------------------------------------------------------
os.environ['NUMBA_CUDA_DRIVER'] = '/usr/lib/x86_64-linux-gnu/libcuda.so'
os.environ['NUMBA_CUDA_LIBDEVICE'] = '/home/zack/Programming/.dask-venv/lib/python3.14/site-packages/nvidia/cuda_nvcc/nvvm/libdevice'

import numba.cuda.cudadrv.nvvm
def mock_get_arch_option(major, minor): 
    return 'compute_80'
numba.cuda.cudadrv.nvvm.get_arch_option = mock_get_arch_option

class MockLibDevice(object):
    def __init__(self):
        self.path = '/home/zack/Programming/.dask-venv/lib/python3.14/site-packages/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc'
    def get(self): 
        with open(self.path, 'rb') as f: return f.read()
numba.cuda.cudadrv.nvvm.LibDevice = MockLibDevice

# ---------------------------------------------------------------------------
# DISPLAY WRAPPER: Terminal colour highlighting
# ---------------------------------------------------------------------------
def colored_pi(true, calc):
    result = ""
    diff = False
    for t, c in zip(true, calc):
        if t == c and not diff:
            result += f"\033[92m{c}\033[0m"   # green
        else:
            diff = True
            result += f"\033[0m{c}"
    if len(calc) > len(true):
        result += f"\033[91m{calc[len(true):]}\033[0m"
    return result

# ---------------------------------------------------------------------------
# MODULE 1: THE REMOTE CPU ENGINE (Numba Parallel Register Loop)
# ---------------------------------------------------------------------------
@njit("uint64[:](uint64, uint64, uint64)", parallel=True, nogil=True)
def run_remote_cpu_simulation(darts_to_throw, wave_count, node_prime_id):
    inside_count = uint64(0)
    for i in prange(darts_to_throw):
        unique_i = (uint64(i) + (uint64(wave_count) * uint64(50001))) * uint64(node_prime_id)
        seed_x = (uint64(6364136223846793005) * unique_i + uint64(1))
        seed_y = (uint64(6364136223846793005) * seed_x + uint64(1))
        x = seed_x / 18446744073709551615.0
        y = seed_y / 18446744073709551615.0
        if (x**2 + y**2) < 1.0:
            inside_count += 1
            
    results_array = np.zeros(2, dtype=np.uint64)
    results_array[0] = inside_count
    results_array[1] = darts_to_throw
    return results_array

# ---------------------------------------------------------------------------
# MODULE 2: THE LOCAL GPU ENGINE (CuPy Register-Locked C++ CUDA Kernel)
# ---------------------------------------------------------------------------
cuda_kernel_code = r'''
extern "C" __global__
void pure_gpu_dart_simulation(const unsigned long long darts_per_thread, const unsigned long long seed_modifier, unsigned int* d_inside_counts) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long state = (unsigned long long)(tx + 1) * (seed_modifier + 9973ULL);
    unsigned int inside_count = 0;
    
    for (unsigned long long i = 0; i < darts_per_thread; i++) {
        state = 6364136223846793005ULL * state + 1442695040888963407ULL;
        double x = (double)state / 18446744073709551615.0;
        
        state = 6364136223846793005ULL * state + 1442695040888963407ULL;
        double y = (double)state / 18446744073709551615.0;
        
        if ((x*x + y*y) < 1.0) { inside_count++; }
    }
    d_inside_counts[tx] = inside_count;
}
'''

def run_local_gpu_wave(total_gpu_darts, wave_seed):
    THREADS_PER_BLOCK = 256
    BLOCKS_PER_GRID = 4096
    TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_PER_GRID
    
    darts_per_thread = int(total_gpu_darts // TOTAL_THREADS)
    if darts_per_thread == 0: darts_per_thread = 1
    actual_gpu_darts = darts_per_thread * TOTAL_THREADS
    
    module = cp.RawModule(code=cuda_kernel_code)
    gpu_func = module.get_function('pure_gpu_dart_simulation')
    
    d_inside_counts = cp.zeros(TOTAL_THREADS, dtype=cp.uint32)
    gpu_func(
        grid=(BLOCKS_PER_GRID,), 
        block=(THREADS_PER_BLOCK,), 
        args=(cp.uint64(darts_per_thread), cp.uint64(wave_seed), d_inside_counts)
    )
    return int(cp.sum(d_inside_counts)), int(actual_gpu_darts)

# ---------------------------------------------------------------------------
# MAIN PROGRAM PIPELINE
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    c = Client("tcp://localhost:8786")
    workers_dict = c.scheduler_info()['workers']
    all_worker_names = list(workers_dict.keys())
    
    remote_cpu_workers = [w for w in all_worker_names if "192.168.1.61" not in w and "127.0.0.1" not in w and "localhost" not in w]
    
    print("Launching Massive 30-Second Balanced Heterogeneous Hybrid Grid...")
    print(f" -> Active Remote CPU Nodes: {remote_cpu_workers}")
    print(" -> Local Master GPU 64-Bit Accelerator Engaged (Master CPU Isolated).")

    strTruePi = "3.14159265358979323846"
    wave_count = 0
    insideCumulated = uint64(0)
    total_real_darts_thrown = uint64(0)
    t0 = None

    cum_w01_darts = uint64(0)
    cum_w02_darts = uint64(0)
    cum_master_cpu_darts = uint64(0)
    cum_master_gpu_darts = uint64(0)

    # 🟢 FIXED: Massive targets to slow down output to a steady 30-second cadence!
    GPU_DARTS_PER_WAVE = 200_000_000_000     # 200 Billion darts per wave for the GPU
    CPU_DARTS_PER_REMOTE_NODE = 2_000_000_000 # 2 Billion darts per wave for w01 and w02

    node_prime_map = {name: uint64(7331 if idx == 0 else 8191) for idx, name in enumerate(remote_cpu_workers)}
    print("-" * 75)

    while True:
        wave_count += 1
        
        if t0 is None:
            t0 = time.time()
            
        # STEP 1: Launch remote CPU tasks
        cpu_futures = []
        for name in remote_cpu_workers:
            prime_id = node_prime_map[name]
            f = c.submit(run_remote_cpu_simulation, uint64(CPU_DARTS_PER_REMOTE_NODE), uint64(wave_count), prime_id, workers=name, pure=False)
            cpu_futures.append(f)

        # STEP 2: Launch local GPU tasks in parallel
        gpu_seed_modifier = uint64(wave_count * 1234567)
        gpu_hits, actual_gpu_darts = run_local_gpu_wave(GPU_DARTS_PER_WAVE, gpu_seed_modifier)
        
        insideCumulated += uint64(gpu_hits)
        total_real_darts_thrown += uint64(actual_gpu_darts)
        cum_master_gpu_darts += uint64(actual_gpu_darts)

        # STEP 3: Gather CPU results and increment counters
        cpu_results = c.gather(cpu_futures)
        for idx, res in enumerate(cpu_results):
            insideCumulated += uint64(res[0])
            total_real_darts_thrown += uint64(res[1])
            
            if idx == 0:
                cum_w01_darts += uint64(res[1])
            elif idx == 1:
                cum_w02_darts += uint64(res[1])

        total_elapsed_time = time.time() - t0
        strCalcPi = f"{4 * insideCumulated / total_real_darts_thrown:.20f}"
        continuous_rate = total_real_darts_thrown / total_elapsed_time / 1_000_000_000      
        
        print(f"Wave: {wave_count} | Calc π = {colored_pi(strTruePi, strCalcPi)}")
        print(f"Continuous Hybrid Cluster Rate: {continuous_rate:,.2f} billion darts/s")
        print(f"Total Active Time: {total_elapsed_time:.2f}s")
        
        print(f"Node Contribution (Cumulative): "
              f"w01: {cum_w01_darts/1e9:.2f}B | "
              f"w02: {cum_w02_darts/1e9:.2f}B | "
              f"master-cpu: {cum_master_cpu_darts/1e9:.2f}B | "
              f"master-gpu: {cum_master_gpu_darts/1e9:.2f}B")
        
        print(f"Grand Total Darts Across CPU+GPU: {total_real_darts_thrown:,}")
        print("-" * 75)
