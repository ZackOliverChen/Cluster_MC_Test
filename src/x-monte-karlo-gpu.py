import time
import cupy as cp

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

if __name__ == '__main__':
    strTruePi = "3.14159265358979323846"
    
    # 1. FIXED: Safer chunk targets for 12GB laptop graphics cards
    DARTS_PER_CHUNK =   500_000_000  # Takes exactly 2GB of VRAM per array
    CHUNKS_PER_WAVE = 20             # Total Wave Target = 10 Billion darts
    TOTAL_WAVE_DARTS = DARTS_PER_CHUNK * CHUNKS_PER_WAVE

    print("Launching Memory-Safe CuPy GPU Matrix...")
    print(f"Darts Per Wave: {TOTAL_WAVE_DARTS:,} ({CHUNKS_PER_WAVE} chunks of {DARTS_PER_CHUNK:,})")
    print("-" * 75)

    wave_count = 0
    insideCumulated = 0
    total_real_darts_thrown = 0
    t0 = None

    rng = cp.random.default_rng(seed=12345)

    while True:
        wave_count += 1
        wave_hits = 0
        
        if t0 is None:
            t0 = time.time()

        for chunk in range(CHUNKS_PER_WAVE):
            # Generate coordinates inside safe 2GB memory pools
            x = rng.random(DARTS_PER_CHUNK, dtype=cp.float32)
            y = rng.random(DARTS_PER_CHUNK, dtype=cp.float32)
            
            # 2. FIXED: In-place mathematical modifications
            # We rewrite the arrays in memory instead of creating new ones!
            cp.multiply(x, x, out=x)  # x now holds x^2
            cp.multiply(y, y, out=y)  # y now holds y^2
            cp.add(x, y, out=x)       # x now holds (x^2 + y^2)
            
            # Create our evaluation mask directly from the output container
            inside_mask = x < 1.0
            
            # Sum up hits for this chunk and clear array memory
            wave_hits += int(cp.sum(inside_mask))
            
            # Explicitly dump blocks to keep usage at baseline levels
            del x, y, inside_mask
            cp.get_default_memory_pool().free_all_blocks()

        print(f"DEBUG: Verified Real Wave Hits = {wave_hits:,}")

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
