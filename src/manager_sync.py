from manager import *
from worker_proc import pure_cpu_dart_simulation, pure_gpu_dart_simulation

class TaskManagerSync(TaskManager):
    def __init__(self, exit_precision=None, exit_time=None):
        super().__init__(exit_precision, exit_time)
        self.wave_count = 0 

    def assign_wave_task(self):
        self.wave_count += 1

        futures = self.assign_full_stack_tasks(pure_cpu_dart_simulation, pure_gpu_dart_simulation)

        if self.t0 is None:
            self.t0 = time.time()
        return futures

    def gather_results(self, futures):
        wave_results = self.client.gather(futures)
        worker_max_times = {name: 0 for name in self.worker_names}

        for res_tuple in wave_results:
            res, worker_name = res_tuple
            self.deal_one_result(res, worker_name)
            if res[2] > worker_max_times[worker_name]: worker_max_times[worker_name] = res[2]
    
        self.total_elapsed_time = time.time() - self.t0
        fCalcPi = 4 * self.insideCumulated / self.total_darts_thrown
        self.convergence_history.append((self.total_darts_thrown, self.total_elapsed_time, fCalcPi))
        self.print_calc_progress(worker_max_times)
        self.tune_cluster_workload(worker_max_times)
        for name, _ in self.worker_task_counts_new.items():
            self.worker_task_counts_new[name] = 0

    def print_calc_progress(self, worker_max_times):
        print(f"{self.wave_count:05d}   ", end = "", flush=True)
        finish = super().print_calc_progress()
        for name in self.worker_names:
            print(f"{worker_max_times[name]/1.0e6:>9.3f} ms", end=name!=self.worker_names[-1] and " | " or "  longest task\n")
        print("-" * 100)
        return finish

    def plot_convergence_history(self):
        super().plot_convergence_history(False)
        