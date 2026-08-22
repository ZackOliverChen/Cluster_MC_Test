from manager import *
from dask.distributed import as_completed
from worker_proc import pure_cpu_dart_simulation, pure_gpu_dart_simulation

class TaskManagerAsync(TaskManager):
    def __init__(self, exit_precision=None, exit_time=None):
        super().__init__(exit_precision, exit_time)
        self.futures = []                   # hold onging futures
        self.future_to_worker_key = {}      # dict to log the key for each ongoing future 
        self.printcycles = 0


    def assign_init_tasks(self):
        task_idx = 0
        for key, name, threads, type in zip(self.worker_keys, self.worker_names, self.worker_nthreads, self.worker_types):
            darts_per_thread_for_this_worker = self.darts_per_thread[name]
            func = pure_gpu_dart_simulation if type == "GPU" else pure_cpu_dart_simulation
            resources = {"GPU": 1} if type == "GPU" else None
            for _ in range(threads):
                seed_x = uint64(self.printcycles *10000 + task_idx)
                seed_y = uint64(self.printcycles *10000 + task_idx + 50000)
                task_idx += 1
                future = self.client.submit(
                    func, 
                    darts_per_thread_for_this_worker, 
                    self.current_offset,
                    seed_x,
                    seed_y,
                    workers=[key], 
                    allow_other_workers=False,
                    pure=False, resources=resources
                )
                self.current_offset += darts_per_thread_for_this_worker
                self.futures.append(future)
                self.future_to_worker_key[future.key] = key # Remember who owns this future
        if self.t0 is None:
            self.t0 = time.time()
        return task_idx


    def loop_check_complete_and_assign_new_tasks(self):
        task_stream = as_completed(self.futures, raise_errors=True)
        worker_max_times = {name: 0 for name in self.worker_names}
        task_idx = 0

        for completed_future in task_stream:
            res, worker_name = completed_future.result()
            self.deal_one_result(res, worker_name)
            if res[2] > worker_max_times[worker_name]: worker_max_times[worker_name] = res[2]

            freed_worker_key = self.future_to_worker_key.pop(completed_future.key)
            darts_per_thread_for_this_worker = self.darts_per_thread[worker_name]
            type = self.worker_types[self.worker_names.index(worker_name)]
            func = pure_gpu_dart_simulation if type == "GPU" else pure_cpu_dart_simulation
            resources = {"GPU": 1} if type == "GPU" else None
            seed_x = uint64(self.printcycles *10000 + task_idx)
            seed_y = uint64(self.printcycles *10000 + task_idx + 50000)
            task_idx += 1
            new_future = self.client.submit(
                func, 
                darts_per_thread_for_this_worker, 
                self.current_offset,
                seed_x,
                seed_y,
                workers=[freed_worker_key], # Send it back to the exact worker that just finished
                allow_other_workers=False,
                pure=False, resources=resources
            )
            self.current_offset += darts_per_thread_for_this_worker
            self.future_to_worker_key[new_future.key] = freed_worker_key
            task_stream.add(new_future)

            total_elapsed_time = time.time() - self.t0
            if total_elapsed_time - self.total_elapsed_time >= self.TARGET_DURATION_NS/1.0e9:
                self.total_elapsed_time = total_elapsed_time
                fCalcPi = 4 * self.insideCumulated / self.total_darts_thrown
                self.convergence_history.append((self.total_darts_thrown, self.total_elapsed_time, fCalcPi))
                self.print_calc_progress()
                self.tune_cluster_workload(None)
                for k in worker_max_times: worker_max_times[k] = 0
                for name, _ in self.worker_task_counts_new.items():
                    self.worker_task_counts_new[name] = 0
                if self.check_finish():
                    print("Preet tasks completed or reached the time limit.")
                    self.CloseClient()
                    return


    def print_calc_progress(self):
        self.printcycles += 1
        print(f"{self.printcycles:05d}   ", end = "", flush=True)
        finish = super().print_calc_progress()
        print("-" * 100)
        return finish
            

    def plot_convergence_history(self):
        super().plot_convergence_history(True)

