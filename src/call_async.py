from manager_async import TaskManagerAsync

def monte_carlo_async(exit_precision=0, exit_time=0):
    tm = TaskManagerAsync(exit_precision, exit_time)   

    tm.CreateClient("tcp://localhost:8786", direct_to_workers=False)

    print(f"\nBegin calculating over the cluster ...")
    print("-" * 100)

    try:
        tm.assign_init_tasks()

        tm.loop_check_complete_and_assign_new_tasks()

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Stopping simulation...")
        tm.CloseClient()

    tm.plot_convergence_history()    

    print("[+] Calculation finished. Goodbye.")
