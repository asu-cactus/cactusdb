import psutil
import subprocess
import time

def get_container_memory_cgroups():
    """
    Get memory information from cgroups (Docker's native way)
    Returns memory usage and limits in bytes
    """
    try:
        # Memory usage from cgroups
        with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
            memory_usage = int(f.read().strip())
            
        # Memory limit from cgroups
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            memory_limit = int(f.read().strip())
            
        # Available memory
        available_memory = memory_limit - memory_usage
            
        return {
            'memory_usage': memory_usage,
            'memory_limit': memory_limit,
            'available_memory': available_memory
        }
    except FileNotFoundError:
        return None

def parse_top_output():
    """
    Run top command and parse its output to get memory information
    Returns dictionary with memory metrics
    """
    try:
        # Run top command in batch mode (-b) and get only one iteration (-n 1)
        cmd = ['top', '-b', '-n', '1']
        output = subprocess.check_output(cmd, universal_newlines=True)
        
        # Parse the output
        memory_info = {}
        for line in output.split('\n'):
            if 'MiB Mem' in line or 'KiB Mem' in line:
                # Remove multiple spaces and split
                parts = ' '.join(line.split()).split()
                
                # Find memory values by looking for numeric values
                for i, part in enumerate(parts):
                    try:
                        if 'total' in parts[i+1].lower():
                            memory_info['total'] = float(part)
                        elif 'free' in parts[i+1].lower():
                            memory_info['free'] = float(part)
                        elif 'used' in parts[i+1].lower():
                            memory_info['used'] = float(part)
                        elif any(x in parts[i+1].lower() for x in ['buff', 'cache', 'buff/cache']):
                            memory_info['buff_cache'] = float(part)
                    except (ValueError, IndexError):
                        continue
                        
        return memory_info if memory_info else None
    except subprocess.CalledProcessError as e:
        print(f"Error running top command: {e}")
        return None

def get_memory_info():
    """
    Get memory information using multiple methods
    Returns consolidated memory information
    """
    memory_info = {
        'cgroups': get_container_memory_cgroups(),
        'top': parse_top_output(),
        'psutil': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'used': psutil.virtual_memory().used,
            'free': psutil.virtual_memory().free
        }
    }
    
    return memory_info

def format_bytes(bytes_value):
    """
    Convert bytes to human readable format
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.2f} TB"

def kill_process_with_most_memory():
    # Get all running processes
    all_processes = [(p.pid, p.name, p.memory_info().rss) for p in psutil.process_iter(['pid', 'name', 'memory_info'])]

    # Sort processes by memory usage in descending order
    all_processes.sort(key=lambda x: x[2], reverse=True)

    # Kill the process using the most memory
    pid_to_kill = all_processes[0][0]
    name_to_kill = all_processes[0][1]
    memory_info = all_processes[0][2]
    subprocess.run(['kill', '-9', str(pid_to_kill)])
    print(f"[INFO] Killed process, PID: {pid_to_kill}, Name: {name_to_kill}, Used Memory: {memory_info}")

def main():
    threshold = 5000 * 1024 * 1024  # 5000 MB threshold in bytes

    while True:
        # available_memory = psutil.virtual_memory().available + psutil.swap_memory().free
        memory_info = get_memory_info()
        available_memory = memory_info['psutil']['available']
        if available_memory < threshold:
            print(f"[INFO] Available memory ({available_memory / (1024 * 1024):.2f} MB) is below threshold {threshold / (1024*1024) :.2f} MB. Killing process with most memory...")
            while psutil.virtual_memory().available < threshold:
                kill_process_with_most_memory()
                time.sleep(0.5)
        else:
            print(f"[INFO] Available memory ({available_memory / (1024 * 1024):.2f} MB) is above threshold {threshold / (1024*1024) :.2f} MB.")

        # Check memory every 1 minute
        time.sleep(5)

if __name__ == "__main__":
    main()
