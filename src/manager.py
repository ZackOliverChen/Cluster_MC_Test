import time
from datetime import timedelta
from dask.distributed import Client
import dask
from numba import uint64
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
from distributed.diagnostics.plugin import UploadDirectory

from dask_cuda import LocalCUDACluster

from globals import TRUE_PI, STR_TRUE_PI, RANDOM_METHOD
from globals import STRATIFIED_SAMPLING, ANTITHETIC_VARIATES
from tools import *

INIT_DARTS_PER_THREAD = uint64(1_000_000_000) 

class TaskManager:
    def __init__(self, exit_precision=None, exit_time=None):
        self.client = None
        self.worker_keys_to_names = {}
        self.worker_names_to_kesys = {}
        self.worker_names = []                # The registered worker name (human readable) 
        self.worker_keys = []                 # Key (addr) for each worker 
        self.worker_nthreads = []             # Number of threads for each worker
        self.worker_types = []                # "GPU" or "CPU"
        #names, keys, nthreads and types must be aligned strictly
        self.convergence_history = []         # [total_darts_thrown, total_elapsed_time, fCalcPi ]
        self.insideCumulated = uint64(0)      # dart number inside the circle
        self.total_darts_thrown = uint64(0)   # total darts thrown
        self.current_offset = uint64(0)       # track task possition to be assigned 
        self.total_elapsed_time = 0.0         # in seconds
        self.t0 = None                        # Time calculation really begin
        self.exit_precision = exit_precision  # calculation stops at this precision (digit number)
        self.exit_time = exit_time            # calculation stops at thie time (in seconds)
        self.worker_task_counts = {}          # tasks completed by each worker (name: count)
        self.worker_task_counts_new = {}      # tasks newly completed, to be shown (name : count)
        
        self.TARGET_DURATION_NS = 10.0 * 1_000_000_000   # every machine to spend roughly x seconds calculating 
        self.darts_per_thread = {}            # (name: darts)
    
    def CreateClient(self, scheduler_address, direct_to_workers=False):
        # Turn off aggressive scheduling theft system-wide
        dask.config.set({'distributed.scheduler.worker-stealing': False})

        # Connect client with centralized data collection routing
        self.client = Client("tcp://localhost:8786", direct_to_workers=False)
#        self.gpu_cluster = LocalCUDACluster(
#            scheduler_address=self.client.scheduler.address,
#            resources={"GPU": 1} 
#        )
#        time.sleep(1.5)   
#        self.client.register_plugin(UploadDirectory("FirstTestProject/src/", update_path=True))     
        for file in [ "src/globals.py", "src/worker_proc.py"]:
            self.client.upload_file(file)

        workers_dict = self.client.scheduler_info()['workers']
        nWorkers = len(workers_dict)
        print(f"Detected Workers: {nWorkers}")

        self.worker_keys = list(workers_dict.keys())
        self.worker_names = [workers_dict[key]['name'] for key in self.worker_keys]
        self.worker_nthreads = [workers_dict[key]['nthreads'] for key in self.worker_keys]
        self.worker_types = ["GPU" if workers_dict[key].get('resources', {}).get('GPU', 0) > 0 else "CPU" for key in self.worker_keys]
        self.worker_names_to_keys = dict(zip(self.worker_names, self.worker_keys))
        self.worker_keys_to_names = dict(zip(self.worker_keys, self.worker_names))
        
        print("Worker Thread Profiles:")
        for key, name, threads in zip(self.worker_keys, self.worker_names, self.worker_nthreads):
            gpu_resource = workers_dict[key].get('resources', {}).get('GPU', 0)
            self.darts_per_thread[name] = INIT_DARTS_PER_THREAD*50 if gpu_resource > 0 else INIT_DARTS_PER_THREAD 
            worker_type = f"GPU enabled" if gpu_resource > 0 else f"{threads} CPU threads"
            print(f" -> {name} ({key}): {worker_type}")
        self.worker_task_counts = {name: 0 for name in self.worker_names}
        self.worker_task_counts_new = {name: 0 for name in self.worker_names}
    
        return self.client


    def CloseClient(self):
        self.client.close()
        self.client = None
        print(" -> Connection to Dask Scheduler terminated cleanly.")


    def tune_cluster_workload(self, worker_max_times):
        damp = 0.9
        if worker_max_times:
            for name, time in worker_max_times.items():
                if time == 0:
                    factor = 0.7 
                else:
                    factor = self.TARGET_DURATION_NS/time
                    factor *= damp if factor > 1.0 else (1 / damp)
                    if factor > 2.0: factor = 2.0 
                    if factor < 0.5: factor = 0.5
                self.darts_per_thread[name] *= factor
                if self.darts_per_thread[name] < 100 : self.darts_per_thread[name] = 100
        else:   # time info not provided, tune by ttask_completed_in_recent_waveask numbers
            for name, nthreads in zip(self.worker_names, self.worker_nthreads):
                task_completed_in_recent_wave = self.worker_task_counts_new[name]
                if task_completed_in_recent_wave == 0:
                    factor = 0.7
                else:
                    factor = float(task_completed_in_recent_wave)/nthreads/1.3
                    factor *= damp if factor > 1.0 else (1 / damp)
                    if factor > 2.0: factor = 2.0 
                    if factor < 0.5: factor = 0.5
                self.darts_per_thread[name] *= factor
                if self.darts_per_thread[name] < 100 : self.darts_per_thread[name] = 100


    def assign_full_stack_tasks(self, cpu_func, gpu_func):
        cpu_darts, cpu_offsets, cpu_seeds_x, cpu_seeds_y, cpu_worker_keys = [], [], [], [], []
        gpu_darts, gpu_offsets, gpu_seeds_x, gpu_seeds_y, gpu_worker_keys = [], [], [], [], []

        task_idx = 0

        for key, name, threads, type in zip(self.worker_keys, self.worker_names, self.worker_nthreads, self.worker_types):
            darts_per_thread_for_this_worker = self.darts_per_thread[name]
            for _ in range(threads):
                seed_x = uint64(self.wave_count * 100000 + task_idx)
                seed_y = uint64(self.wave_count * 100000 + task_idx + 50000)                
                if type == "GPU":
                    gpu_darts.append(darts_per_thread_for_this_worker)
                    gpu_offsets.append(self.current_offset)
                    gpu_seeds_x.append(seed_x)
                    gpu_seeds_y.append(seed_y)
                    gpu_worker_keys.append(key)
                    self.current_offset += darts_per_thread_for_this_worker
                else:
                    cpu_darts.append(darts_per_thread_for_this_worker)
                    cpu_offsets.append(self.current_offset)
                    cpu_seeds_x.append(seed_x)
                    cpu_seeds_y.append(seed_y)
                    cpu_worker_keys.append(key)
                    self.current_offset += darts_per_thread_for_this_worker
                task_idx += 1

        futures = []
        future_to_worker_key = {}
        # 1. CPU DISPATCH: No resource tags! All 19 threads will saturate instantly at 100%
        if cpu_worker_keys:
            cpu_futures = self.client.map(
                cpu_func, 
                cpu_darts, cpu_offsets, cpu_seeds_x, cpu_seeds_y,
                workers=cpu_worker_keys,
                allow_other_workers=False,
                pure=False
            )
            futures.extend(cpu_futures)

        # 2. GPU DISPATCH: Only tag the GPU function to ensure it locks onto your dask-cuda-workers
        if gpu_worker_keys:
            gpu_futures = self.client.map(
                gpu_func, 
                gpu_darts, gpu_offsets, gpu_seeds_x, gpu_seeds_y,
                workers=gpu_worker_keys,
                allow_other_workers=False,
                pure=False,
                resources={"GPU": 1} # Perfectly safe because your GPU workers are spawned with 1 thread each!
            )
            futures.extend(gpu_futures)
        return futures

    def deal_one_result(self, res, worker_name):
        self.insideCumulated += uint64(res[0])
        self.total_darts_thrown += uint64(res[1])
        self.worker_task_counts[worker_name] += 1
        self.worker_task_counts_new[worker_name] += 1

    def check_finish(self):
        if len(self.convergence_history) == 0: return False
        precision = precision_pi(self.convergence_history[-1][2])
        finish = False
        if self.exit_precision > 0 and precision >= self.exit_precision:
            finish = True
        if self.exit_time > 0 and self.total_elapsed_time >= self.exit_time:
            finish = True
        return finish

    def print_calc_progress(self):
        if len(self.convergence_history) == 0:
            return
        strCalcPi = f"{self.convergence_history[-1][2]:.20f}"
        continuous_rate = self.total_darts_thrown / self.total_elapsed_time / 1_000_000_000
        strcolorpi, precision = colored_pi(strCalcPi)
        print(f"Calc π = {strcolorpi}")
        print(f"Continuous Dart Rate: {continuous_rate:,.2f} billion darts/s, Total:{self.total_darts_thrown / 1.0e9:,.1f} billions")
        duration = timedelta(seconds=int(self.total_elapsed_time))
        milliseconds = int((self.total_elapsed_time - int(self.total_elapsed_time)) * 1_000)
        print(f"Total Elapsed Time: {duration}.{milliseconds:03d}")
        for name, darts in self.worker_task_counts_new.items():
            is_last = (name == self.worker_names[-1])
            print(f"{name:<8}: {darts:>2}", end="  newly completed\n" if is_last else " | ")
        for name, darts in self.worker_task_counts.items():
            is_last = (name == self.worker_names[-1])
            print(f"{darts:>12}", end="  tasks accumulated\n" if is_last else " | ")
        for name, darts in self.darts_per_thread.items():
            is_last = (name == self.worker_names[-1])
            print(f"{darts/1e6:>12.2f}", end="  task size (M)\n" if is_last else " | ")
        return


    def plot_convergence_history(self, async_mode=False):
        def x_axis_format(x, pos):
            hours = int(x // 3600)
            minutes = int((x % 3600) // 60)
            seconds = int(x % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        if len(self.convergence_history) == 0:
            return
        
        print("\n -> Generating multi-axis time-series convergence plot...")
        columns = ["total_darts", "elapsed_time", "calc_pi"]
        df = pd.DataFrame(self.convergence_history, columns=columns)
        
        # Ensure correct data type conversions
        df["calc_pi"] = df["calc_pi"].astype(float)
        df["total_darts"] = df["total_darts"].astype(float)
        df["elapsed_time"] = df["elapsed_time"].astype(float) # In seconds
        
        df["error"] = (df["calc_pi"] - TRUE_PI)
                
        # 1. Initialize the base figure configuration
        fig, ax1 = plt.subplots(figsize=(11, 6), dpi=100)
        
        # 2. Draw Left Axis (ax1): Absolute Error Delta (Log Scale)
        color_err = "Green" # Green
        line1 = ax1.plot(df["elapsed_time"], df["error"], 
                            color=color_err, linestyle="-", marker="o", 
                            linewidth=2, markersize=5, label="Error (Calc - True $\\pi$)")
        ax1.xaxis.set_major_formatter(FuncFormatter(x_axis_format))
        ax1.set_yscale("symlog", linthresh=1e-6) # Use symlog to handle both positive and negative errors
        #ax1.set_yscale("log")
        ax1.set_xlabel("Total Continuous Compute Runtime (HH:MM:SS)", fontsize=11, labelpad=10)
        ax1.set_ylabel("Error Delta from True $\\pi$ (Log Scale)", color=color_err, fontsize=11, labelpad=10)
        ax1.tick_params(axis='y', labelcolor=color_err)
        ax1.grid(True, which="both", linestyle="--", alpha=0.4)

        # 3. Draw Right Axis (ax2): calcPi
        ax2 = ax1.twinx() 
        color_pi = "red" 
        line2 = ax2.plot(df["elapsed_time"], df["calc_pi"], 
                        color=color_pi, linestyle="--", marker="x", 
                        linewidth=1.5, markersize=5, label="Calculated $\\pi$ Value")
        ax2.axhline(y=TRUE_PI, color="black", linestyle=":", linewidth=1.5, label="True $\\pi$ Value")
        ax2.set_ylabel("Calculated $\\pi$ Value", color='none', fontsize=11, labelpad=10)
        ax2.tick_params(axis='y', labelcolor=color_pi)
        # FIX: Create a dedicated formatter that strictly forbids offsets and scientific notation
        pi_formatter = ScalarFormatter(useOffset=False)
        pi_formatter.set_scientific(False)
        ax2.yaxis.set_major_formatter(pi_formatter)

        # 4. Draw the Second Right Axis (ax3): Total Accumulated Darts
        ax3 = ax1.twinx()
        color_darts = "Blue" 
        line3 = ax3.plot(df["elapsed_time"], df["total_darts"], 
                        color=color_darts, linestyle="--", marker="x", 
                        linewidth=1.5, markersize=5, label="Total Accumulated Darts")
        ax3.set_ylabel("Total Accumulated Darts", color=color_darts, fontsize=11, labelpad=10)
        ax3.tick_params(axis='y', labelcolor=color_darts)
        # Push the third axis spine 60 pixels to the right
        ax3.spines['right'].set_position(('outward', 60))
        # Format the right y-axis numbers nicely into Billions (B) or Millions (M)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e9:,.1f}B" if x >= 1e9 else f"{x/1e6:,.1f}M"))

        target_width = precision_pi(df["calc_pi"].iloc[-1])
        strFinalPi = f"{df['calc_pi'].iloc[-1]:<.20f}"[:target_width]
        strPIs = fr"Target $\pi$: {STR_TRUE_PI}   Finnal Calculated: {strFinalPi}"
        ax2.text(0.02, TRUE_PI, strPIs, color="black", va="bottom", ha="left", transform=ax2.get_yaxis_transform())

        # 5. Consolidate and merge the legends smoothly from all tracks
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=True)
        if self.worker_nthreads is not None:
            CLUSTER_INFO = f"{len(self.worker_nthreads)} workers, {sum(self.worker_nthreads)} cores"
        else:
            CLUSTER_INFO = f""
        strRandomMethod = RANDOM_METHOD
        if STRATIFIED_SAMPLING: strRandomMethod += " S"
        if ANTITHETIC_VARIATES: strRandomMethod += " A"
        strTaskMode = " Async" if async_mode else "Sync"
        ax1.set_title(f"$\\pi$ Monte-Carlo ({strRandomMethod}, {CLUSTER_INFO}, {strTaskMode})", fontsize=13, fontweight="bold", pad=15)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        plt.figtext(0.98, 0.02, f"Generated: {timestamp}", ha='right', va='bottom', fontsize=10, color='lightgray')

        plt.tight_layout()
        if len(self.convergence_history) >= 10:
            filename = f"pi_converg_{time.strftime('%y%m%d_%H%M')}.png"
        else:
            filename = f"pi_converg.png"
        plt.savefig(filename, bbox_inches="tight")
        print(f"[+] Performance profile chart saved successfully as '{filename}'.")
        try:
            display(fig)
        except NameError:
            plt.show()


