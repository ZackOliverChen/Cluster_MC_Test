import time
import cupy as cp

#Please run this script from shell using the following command:
# LD_LIBRARY_PATH=/home/zack/Programming/.dask-venv/lib /home/zack/Programming/.dask-venv/bin/python ...


# ===========================================================================
# 🟢 TOGGLE SWITCH: Change this to True for 64-bit precision, False for 32-bit speed
USE_64BIT = True  
# ===========================================================================

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

# 1. FIXED: Added C++ Macros to dynamically rewrite the data types at compilation time
cuda_kernel_code = f'''
#define PRECISION_64BIT {1 if USE_64BIT else 0}

#if PRECISION_64BIT
    typedef unsigned long long uint_t;
    typedef double             float_t;
    #define LCG_MULT           6364136223846793005ULL
    #define LCG_INC            1442695040888963407ULL
    #define SCALE_MAX          18446744073709551615.0
#else
    typedef unsigned int       uint_t;
    typedef float              float_t;
    #define LCG_MULT           1664525U
    #define LCG_INC            1013904223U
    #define SCALE_MAX          4294967295.0f
#endif

extern "C" __global__
void pure_gpu_dart_simulation(const uint_t darts_per_thread, const uint_t seed_modifier, unsigned int* d_inside_counts) {{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Initialize state using the macro type definition
    uint_t state = (uint_t)(tx + 1) * (seed_modifier + (uint_t)7331);
    unsigned int inside_count = 0;
    
    for (uint_t i = 0; i < darts_per_thread; i++) {{
        // The loop layout remains completely unchanged! The macros map the logic automatically.
        state = LCG_MULT * state + LCG_INC;
        float_t x = (float_t)state / SCALE_MAX;
        
        state = LCG_MULT * state + LCG_INC;
        float_t y = (float_t)state / SCALE_MAX;
        
        if ((x*x + y*y) < (float_t)1.0) {{
            inside_count++;
        }}
    }}
    d_inside_counts[tx] = inside_count;
}}
'''

if __name__ == '__main__':
    strTruePi = "3.14159265358979323846"
    
    THREADS_PER_BLOCK = 256  
    BLOCKS_PER_GRID = 4096  
    TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_PER_GRID
    
    DARTS_PER_THREAD = 1_000_000 
    TOTAL_WAVE_DARTS = TOTAL_THREADS * DARTS_PER_THREAD

    precision_string = "64-Bit Precision" if USE_64BIT else "32-Bit Speed"
    print(f"Launching Register-Locked GPU Simulation Matrix ({precision_string})...")
    print(f"Active GPU Hardware Threads: {TOTAL_THREADS:,}")
    print(f"Darts Per Wave: {TOTAL_WAVE_DARTS:,}")
    print("-" * 75)

    # Compile the macro-translated C++ code instantly via CuPy
    module = cp.RawModule(code=cuda_kernel_code)
    pure_gpu_dart_simulation = module.get_function('pure_gpu_dart_simulation')

    wave_count = 0
    insideCumulated = 0
    total_real_darts_thrown = 0
    t0 = None

    # Pick the correct array data type to pass down the python arguments bridge
    arg_type = cp.uint64 if USE_64BIT else cp.uint32

    while True:
        wave_count += 1
        seed_modifier = int(wave_count * 2862933)
        
        d_inside_counts = cp.zeros(TOTAL_THREADS, dtype=cp.uint32)
        
        if t0 is None:
            t0 = time.time()
            
        pure_gpu_dart_simulation(
            grid=(BLOCKS_PER_GRID,), 
            block=(THREADS_PER_BLOCK,), 
            args=(arg_type(DARTS_PER_THREAD), arg_type(seed_modifier), d_inside_counts)
        )
        
        wave_hits = int(cp.sum(d_inside_counts))
        
        insideCumulated += wave_hits
        total_real_darts_thrown += TOTAL_WAVE_DARTS
        
        total_elapsed_time = time.time() - t0
        strCalcPi = f"{4 * insideCumulated / total_real_darts_thrown:.20f}"
        continuous_rate = total_real_darts_thrown / total_elapsed_time / 1_000_000_000      
        
        print(f"Wave: {wave_count} | Calc π = {colored_pi(strTruePi, strCalcPi)}")
        print(f"Continuous GPU Rate: {continuous_rate:,.2f} billion darts/s")
        print(f"Total Active Time: {total_elapsed_time:.2f}s")
        print(f"Grand Total Darts: {total_real_darts_thrown:,}")
        print("-" * 75)
