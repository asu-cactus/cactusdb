import subprocess
import time

# Interval between each check (in seconds)
interval = 3

# Output file
output_file = "cpu_usage.log"

# Function to get top CPU process and usage
def get_top_cpu():
    output = subprocess.run(['top', '-b', '-n', '1'], capture_output=True, text=True)
    lines = output.stdout.split('\n')
    cpu_line = [line for line in lines if '%Cpu(s):' in line][0]
    top_process = lines[7]
    return cpu_line, top_process

# Main loop
with open(output_file, 'w') as f:
    try:
        while True:
            cpu_line, top_process = get_top_cpu()
            f.write(f"CPU Usage: {cpu_line.split()[1]}%\n")
            f.write("Top Process:\n")
            f.write(f"{top_process}\n")
            f.write("\n")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Script stopped manually.")