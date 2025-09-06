#!/usr/bin/env python3
import subprocess
import os

def main():
    # List of batch sizes
    # batch_sizes = [2048, 4196, 5000, 10000]
    drivers = [4,6,8,10,12]

    # Correct full path to the binary (note: no `.` at start)
    binary_path = "/home/cactusdb/_build/release/velox/optimizer/tests/profile_query_generator"

    # Ensure log directory exists
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    # Loop through batch sizes and run command
    for size in drivers:
        log_file = os.path.join(log_dir, f"expedia_driver_{size}.log")
        cmd = [
            binary_path,
            "--workload=imbridge",
            "--query_template=expedia",
            f" --data_batch_size=1024"
            f"--num_driver={size}"
        ]

        print(f"Running batch_size={size} → logging to {log_file}")
        with open(log_file, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            ret = proc.wait()
            if ret != 0:
                print(f"⚠️  Failed: batch_size={size}, exit code {ret}")

    print("✅ All batch sizes completed.")

if __name__ == "__main__":
    main()
