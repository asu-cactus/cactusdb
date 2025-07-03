# TODO: needs to be merged
import subprocess
import time
import re
import csv
import logging
from datetime import datetime
from tqdm import tqdm

# Configure logging for errors
logging.basicConfig(
    filename="execution_errors.log",
    filemode="w",
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Path to the binary
binary = "/home/cactusdb/_build/release/velox/optimizer/tests/profile_query_generator"

# Workload-template combinations
workload_template_pairs = [
    ("movielens1", "template4"),
    ("movielens1", "template8"),
    ("movielens1", "template9"),
    ("tpcxai1",    "template5"),
    ("tpcxai1",    "template10"),
    ("tpcxai1",    "template9"),
]

# Number of repetitions per command
repetitions = 30

# Output CSV file
csv_file = "query_results.csv"

# Regex to extract reported execution time
time_pattern = re.compile(r"\[INFO\] Execution time:\s*([\d.]+)")

# Prepare CSV
with open(csv_file, mode="w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["command", "execution_time", "success"]
    )
    writer.writeheader()

    # Flatten all runs into a list for a single progress bar
    runs = [
        (workload, template, i)
        for workload, template in workload_template_pairs
        for i in range(1, repetitions + 1)
    ]

    # Iterate with tqdm progress bar
    for workload, template, run_id in tqdm(runs, desc="Total runs", unit="run"):
        # Build command
        cmd = (
            f"{binary} "
            f"-workload={workload} "
            f"-query_template={template} "
            "-verbose=4"
        )

        # Extract parameters string
        params = " ".join(tok for tok in cmd.split() if tok.startswith("-"))

        start = time.time()
        try:
            proc = subprocess.run(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            elapsed = time.time() - start

            stdout = proc.stdout
            stderr = proc.stderr

            # Determine success
            segfault = "segmentation fault" in stderr.lower()
            success = not segfault and proc.returncode == 0

            # Parse reported execution time
            match = time_pattern.search(stdout)
            reported_time = float(match.group(1)) if match else None

            execution_time = reported_time if reported_time is not None else elapsed

            # Write to CSV
            writer.writerow({
                "command": params,
                "execution_time": f"{execution_time:.6f}",
                "success": success
            })

            # Log errors
            if not success:
                logging.error(
                    f"Run {run_id} | Workload: {workload} | Template: {template} | "
                    f"Return Code: {proc.returncode} | Segfault: {segfault} | "
                    f"Stderr: {stderr.strip()}"
                )

        except Exception as e:
            logging.error(
                f"Run {run_id} | Workload: {workload} | Template: {template} | Exception: {e}"
            )
            writer.writerow({
                "command": params,
                "execution_time": "",
                "success": False
            })

print("Execution complete. Results saved to query_results.csv and errors to execution_errors.log")
