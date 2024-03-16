import subprocess
import pandas as pd
import numpy as np
import itertools
from tqdm.auto import tqdm


def run_cpp_program(path, params):
    # return 0
    execution_command = [
        "/home/ubuntu/velox/build/velox/optimizer/verticalmul_test {}".format(params)
    ]
    subprocess.run(execution_command, shell=True)



if __name__ == "__main__":
    list_feature_size = [597540]
    list_sample_size = [1, 10, 100, 1000, 10000, 100000]
    list_output_size = [1024]
    list_implements = ["vertical"]
    # list_velox_driver = [1, 2, 4, 8]
    list_velox_driver = [1]
    # list_function_threads = [1, 2, 4, 8]
    list_function_threads = [8]

    run_configs = list(
        itertools.product(
            list_sample_size, list_feature_size, list_output_size, list_velox_driver, list_function_threads, list_implements
        )
    )
    result_df = None
    for config in tqdm(run_configs):
        sample_size, feature_size, output_size, velox_driver, function_threads, implements = config
        params = "{} {} {} {} {} {}".format(
            sample_size, feature_size, output_size, velox_driver, function_threads, implements
        )

        run_cpp_program("/", params)