import psutil
import subprocess
import time

def kill_process_with_most_memory():
    # Get all running processes
    all_processes = [(p.pid, p.memory_info().rss) for p in psutil.process_iter(['pid', 'name', 'memory_info'])]

    # Sort processes by memory usage in descending order
    all_processes.sort(key=lambda x: x[1], reverse=True)

    # Kill the process using the most memory
    pid_to_kill = all_processes[0][0]
    name_to_kill = all_processes[0][1]
    memory_info = all_processes[0][2]
    subprocess.run(['kill', '-9', str(pid_to_kill)])
    print(f"[INFO] Killed process, PID: {pid_to_kill}, Name: {name_to_kill}, Used Memory: {memory_info}")

def main():
    threshold = 2000 * 1024 * 1024  # 2000 MB threshold in bytes

    while True:
        available_memory = psutil.virtual_memory().available
        if available_memory < threshold:
            print(f"[INFO] Available memory ({available_memory / (1024 * 1024):.2f} MB) is below threshold. Killing process with most memory...")
            kill_process_with_most_memory()
        else:
            print(f"[INFO] Available memory ({available_memory / (1024 * 1024):.2f} MB) is above threshold.")

        # Check memory every 1 minute
        time.sleep(10)

if __name__ == "__main__":
    main()
