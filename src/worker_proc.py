from dask.distributed import get_worker
import time
import numpy as np
from numba import njit, prange, uint64, cuda
from globals import RANDOM_METHOD, NUMBA_PARALLEL
#import cupy as cp


# Core computational block running inside raw machine registers (Built-in Numba Engine)
@njit(nogil=True, cache=False, parallel=NUMBA_PARALLEL)
def _pure_cpu_dart_simulation_numpy(darts_to_throw, task_offset, wave_seed_x, wave_seed_y):
    inside_count = uint64(0)

    # 1. Generate a unified, isolated seed value combining inputs
    # This guarantees every thread on every worker gets a unique starting track
    combined_seed = uint64(task_offset) + uint64(wave_seed_x) + uint64(wave_seed_y)
    
    # 2. Seed Numba's internal thread-safe random state engine
    # (Converting to a 32-bit uint array mask makes it safe across all platforms)
    np.random.seed(int(combined_seed & uint64(0xFFFFFFFF)))

    # 3. Fast, optimized execution loop
    for i in prange(darts_to_throw):  #prange defaults to a dequential loop if NUMBA_PARALLEL is False
        # Uses Numba's highly optimized, statistically perfect native random float generator
        x = np.random.random()
        y = np.random.random()
        
        if (x*x + y*y) < 1.0:
            inside_count += 1
            
    # 4. Correctly assign elements to array slots via indices
    results_array = np.zeros(3, dtype=np.uint64)
    results_array[0] = inside_count  
    results_array[1] = darts_to_throw 
    results_array[2] = uint64(0)  
    return results_array


# Core computational block running inside raw machine registers
@njit("uint64[:](uint64, uint64, uint64, uint64)", nogil=True, cache=False, parallel=NUMBA_PARALLEL)
def _pure_cpu_dart_simulation_xorshift64(darts_to_throw, task_offset, wave_seed_x, wave_seed_y):
    inside_count = uint64(0)

    # XORSHIFT64 randomizer that mixes indices, worker offsets, and wave seeds
    state_x = (uint64(task_offset) + wave_seed_x + uint64(1)) * uint64(7331)
    #state_y = (uint64(task_offset) + wave_seed_y + uint64(1)) * uint64(7331)
    for i in prange(darts_to_throw):        

        # Throw X coordinate
        state_x ^= state_x << uint64(13)
        state_x ^= state_x >> uint64(7)
        state_x ^= state_x << uint64(17)
        #x = state_x / 18446744073709551615.0
        
        # Throw Y coordinate
        #state_y ^= state_y << uint64(12)
        #state_y ^= state_y >> uint64(25)
        #state_y ^= state_y << uint64(27)
        #y = state_y / 18446744073709551615.0
                
        x = float(state_x >> 32) * 2.3283064365386963e-10
        y = float(state_x & 0xFFFFFFFF) * 2.3283064365386963e-10

        if (x**2 + y**2) < 1.0:
            inside_count += 1
            
    results_array = np.zeros(3, dtype=np.uint64)
    results_array[0] = inside_count  
    results_array[1] = darts_to_throw 
    results_array[2] = uint64(0)  # Placeholder for elapsed time, unknown yet
    return results_array

# Core computational block running inside raw machine registers (PCG64 engine)
@njit(nogil=True, cache=False, parallel=NUMBA_PARALLEL)
def _pure_cpu_dart_simulation_pcg64(darts_to_throw, task_offset, wave_seed_x, wave_seed_y):
    inside_count = uint64(0)

    # 1. State values for the PCG64 LCG track (128-bit state simulated via two uint64 blocks)
    # X Stream State
    state_x_high = uint64(wave_seed_x)
    state_x_low  = uint64(task_offset) + uint64(1)
    
    # Y Stream State
    state_y_high = uint64(wave_seed_y)
    state_y_low  = uint64(task_offset) + uint64(50001) # Shifted to isolate paths

    # 2. Fixed Multipliers and Increments defined by Melissa O'Neill's PCG specification
    MULT_HIGH = uint64(2549297995355413924)
    MULT_LOW  = uint64(4865540595714422341)
    INC_X     = uint64(1442695040888963407)
    INC_Y     = uint64(8589934592) # Distinct stream increment constant

    for i in prange(darts_to_throw):        
        
        # --- STREAM X: Step 128-bit LCG ---
        # (High * Low cross-multiplication with native 64-bit overflow mechanics)
        carry_x = (state_x_low * MULT_LOW)
        state_x_high = (state_x_high * MULT_LOW) + (state_x_low * MULT_HIGH) + (carry_x >> uint64(63))
        state_x_low = carry_x + INC_X
        
        # Stream X Output Permutation (XSH-RR variant: extra bitwise folding & rotate right)
        x_rot = uint64((state_x_high ^ (state_x_high >> uint64(18))) >> uint64(27))
        rot_amount_x = state_x_high >> uint64(59)
        raw_x = (x_rot >> rot_amount_x) | (x_rot << ((-rot_amount_x) & uint64(31)))
        x = float(raw_x & uint64(0xFFFFFFFF)) / 4294967295.0


        # --- STREAM Y: Step 128-bit LCG ---
        carry_y = (state_y_low * MULT_LOW)
        state_y_high = (state_y_high * MULT_LOW) + (state_y_low * MULT_HIGH) + (carry_y >> uint64(63))
        state_y_low = carry_y + INC_Y
        
        # Stream Y Output Permutation
        y_rot = uint64((state_y_high ^ (state_y_high >> uint64(18))) >> uint64(27))
        rot_amount_y = state_y_high >> uint64(59)
        raw_y = (y_rot >> rot_amount_y) | (y_rot << ((-rot_amount_y) & uint64(31)))
        y = float(raw_y & uint64(0xFFFFFFFF)) / 4294967295.0


        # --- Geometry Evaluation Engine ---
        if (x*x + y*y) < 1.0:
            inside_count += 1
            
    results_array = np.zeros(3, dtype=np.uint64)
    results_array[0] = inside_count  
    results_array[1] = darts_to_throw 
    results_array[2] = uint64(0)  
    return results_array


# Core computational block running inside raw machine registers (Native PCG32 Engine)
@njit(nogil=True, cache=False, parallel=NUMBA_PARALLEL)
def _pure_cpu_dart_simulation_pcg32(darts_to_throw, task_offset, wave_seed_x, wave_seed_y):
    inside_count = uint64(0)

    # 1. Initialize native 64-bit states for X and Y streams
    state_x = uint64(task_offset) + uint64(wave_seed_x) + uint64(1)
    state_y = uint64(task_offset) + uint64(wave_seed_y) + uint64(50001)

    # Standard Melissa O'Neill PCG32 constants (64-bit LCG multiplier + distinct increments)
    PCG_MULT = uint64(6364136223846793005)
    INC_X    = uint64(1442695040888963407)
    INC_Y    = uint64(8589934592)

    for i in prange(darts_to_throw):        
        
        # --- STREAM X: Step State & Apply XSH-RR Permutation ---
        state_x = state_x * PCG_MULT + INC_X
        xorshifted_x = np.uint32(((state_x >> uint64(18)) ^ state_x) >> uint64(27))
        rot_x = np.uint32(state_x >> uint64(59))
        raw_x = (xorshifted_x >> rot_x) | (xorshifted_x << ((-rot_x) & np.uint32(31)))
        x = float(raw_x) / 4294967295.0

        # --- STREAM Y: Step State & Apply XSH-RR Permutation ---
        state_y = state_y * PCG_MULT + INC_Y
        xorshifted_y = np.uint32(((state_y >> uint64(18)) ^ state_y) >> uint64(27))
        rot_y = np.uint32(state_y >> uint64(59))
        raw_y = (xorshifted_y >> rot_y) | (xorshifted_y << ((-rot_y) & np.uint32(31)))
        y = float(raw_y) / 4294967295.0

        # --- Geometry Evaluation Engine ---
        if (x*x + y*y) < 1.0:
            inside_count += 1
            
    # Correctly return array elements via indices
    results_array = np.zeros(3, dtype=np.uint64)
    results_array[0] = inside_count  
    results_array[1] = darts_to_throw 
    results_array[2] = uint64(0)  
    return results_array

def pure_cpu_dart_simulation(darts_to_throw, task_offset, wave_seed_x, wave_seed_y):
    start_time = time.perf_counter()

    if RANDOM_METHOD == "XORSHIFT64":
        results_array = _pure_cpu_dart_simulation_xorshift64(darts_to_throw, task_offset, wave_seed_x, wave_seed_y)
    elif RANDOM_METHOD == "PCG64":
        results_array = _pure_cpu_dart_simulation_pcg64(darts_to_throw, task_offset, wave_seed_x, wave_seed_y)
    elif RANDOM_METHOD == "PCG32":
        results_array = _pure_cpu_dart_simulation_pcg32(darts_to_throw, task_offset, wave_seed_x, wave_seed_y)
    elif RANDOM_METHOD == "NUMPY":
        results_array = _pure_cpu_dart_simulation_numpy(darts_to_throw, task_offset, wave_seed_x, wave_seed_y)
    else:
        raise ValueError(f"Unknown RANDOM_METHOD: {RANDOM_METHOD}") 
    
    elapsed_seconds = time.perf_counter() - start_time
    worker = get_worker()
    results_array[2] = uint64(elapsed_seconds * 1_000_000_000)  # Convert to nanoseconds
    return results_array, worker.name

#===============================================================================================
@cuda.jit
def _pure_gpu_dart_simulation_xorshift64(task_offset, wave_seed_x, wave_seed_y, darts_per_core, out_inside_counts):
    thread_id = cuda.grid(1)

    core_offset = uint64(task_offset) + (uint64(thread_id) * uint64(darts_per_core))

    state_x = (core_offset + uint64(wave_seed_x) + uint64(1)) * uint64(7331)
    #state_y = (core_offset + uint64(wave_seed_y) + uint64(1)) * uint64(7331)
    
    inside_count = uint64(0)
    INV_2_32 = 2.3283064365386963e-10
    
    # 2. This core loops through its own portion of the dart pile
    for _ in range(darts_per_core):        
        # Throw X coordinate
        state_x ^= state_x << uint64(13)
        state_x ^= state_x >> uint64(7)
        state_x ^= state_x << uint64(17)
        #x = state_x / 18446744073709551615.0
        
        # Throw Y coordinate
        #state_y ^= state_y << uint64(12)
        #state_y ^= state_y >> uint64(25)
        #state_y ^= state_y << uint64(27)
        #y = state_y / 18446744073709551615.0

        # Split one 64-bit result into two 32-bit coords
        x = (state_x >> uint64(32))        * INV_2_32   # upper 32 bits
        y = (state_x & uint64(0xFFFFFFFF)) * INV_2_32   # lower 32 bits
        
        if (x*x + y*y) < 1.0:
            inside_count += uint64(1)
            
    # 3. Store this core's unique count safely inside its designated array slot
    out_inside_counts[thread_id] = inside_count

@cuda.reduce
def sum_reduce(a, b):
    return a + b

# ── Cache device info once at import time (runs once per worker process) ──
_DEVICE = None
MAX_THREADS_PER_BLOCK = 0
BLOCKS_PER_GRID = 0
TOTAL_GPU_THREADS = 0
def get_gpu_info():
    global _DEVICE, MAX_THREADS_PER_BLOCK, BLOCKS_PER_GRID, TOTAL_GPU_THREADS
    _DEVICE               = cuda.get_current_device()
    MAX_THREADS_PER_BLOCK = _DEVICE.MAX_THREADS_PER_BLOCK
    NUM_SMS               = _DEVICE.MULTIPROCESSOR_COUNT
    BLOCKS_PER_GRID       = NUM_SMS * 4
    TOTAL_GPU_THREADS     = MAX_THREADS_PER_BLOCK * BLOCKS_PER_GRID

def pure_gpu_dart_simulation(darts_to_throw, task_offset, wave_seed_x, wave_seed_y):
    t_start = time.perf_counter_ns()

    if not _DEVICE: get_gpu_info()
    # Divide total darts evenly across your hardware execution units
    darts_per_core = uint64(darts_to_throw // TOTAL_GPU_THREADS)
    
    # Allocate a temporary results buffer array directly inside the GPU VRAM
    out_inside_counts = cuda.device_array(TOTAL_GPU_THREADS, dtype='uint64')
    
    # Launch the code onto the physical GPU hardware grid
    _pure_gpu_dart_simulation_xorshift64[BLOCKS_PER_GRID, MAX_THREADS_PER_BLOCK](
        task_offset, wave_seed_x, wave_seed_y, darts_per_core, out_inside_counts
    )
        
    # Returns the exact 3-element array your TaskManagerSync.gather_results logic expects
    results_array = np.zeros(3, dtype=np.uint64)
    results_array[0] = np.uint64(sum_reduce(out_inside_counts))
    results_array[1] = uint64(darts_to_throw)
    results_array[2] = uint64(time.perf_counter_ns() - t_start)
    worker = get_worker()
    return results_array, worker.name