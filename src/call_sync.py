from manager_sync import TaskManagerSync

def monte_carlo_sync(exit_precision=0, exit_time=0):
    tm = TaskManagerSync(exit_precision, exit_time)

    tm.CreateClient("tcp://localhost:8786", direct_to_workers=False)

    print(f"\nBegin calculating over the cluster ...")
    print("-" * 100)
    try:
        while True:
            futures = tm.assign_wave_task()
            tm.gather_results(futures)
            if tm.check_finish():
                print("Preset tasks completed or reached the time limit.")
                tm.CloseClient()
                break
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Stopping simulation...")
        tm.CloseClient()
    tm.plot_convergence_history()
    print("[+] Calculation finished. Goodbye.")        
