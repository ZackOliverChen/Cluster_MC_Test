"""
Monte Carlo π approximation using Dask
Calculates π by throwing darts at a square dartboard with a circle 
inscribed in it. The ratio of darts that land inside the circle to 
the total number of darts thrown is used to approximate π.
"""
from dask.distributed import Client
import dask.array as da
import time

def colored_pi(true, calc):
    result = ""
    diff = False
    for t, c in zip(true, calc):
        if t == c and not diff:
            result += f"\033[92m{c}\033[0m"   # green
        else:
            diff = True
            result += f"\033[0m{c}"   # normal text color
    if len(calc) > len(true):
        result += f"\033[91m{calc[len(true):]}\033[0m"
    return result

c = Client("tcp://localhost:8786")
nWorkers = len(c.scheduler_info()['workers'])
print(f"Workers: {nWorkers}")

# 400 million darts across the cluster
N = 400_000_000
chunk = N // nWorkers  # Distribute chunks evenly among workers

strTruePi = "3.14159265358979323846"
strCalcPi = ""
t0 = time.time()

nRuns = 0
insideCumulated = 0

while strCalcPi != strTruePi:
    nRuns += 1
    x = da.random.random(N, chunks=chunk)
    y = da.random.random(N, chunks=chunk)
    inside = (x**2 + y**2) < 1.0

    insideCumulated += inside.sum().compute()
    
    strCalcPi = f"{4 * insideCumulated / (nRuns * N):.20f}"
    t1 = time.time()
    print(f"True π = {strTruePi}")
    print(f"Calc π = {colored_pi(strTruePi, strCalcPi)}")
    print(f"Runs: {nRuns}, Time: {t1-t0:.2f}s, Inside: {insideCumulated}, Total: {nRuns * N}")
c.close()