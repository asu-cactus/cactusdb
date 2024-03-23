import subprocess
import pandas as pd
import numpy as np
import itertools
from tqdm.auto import tqdm


def run_cpp_program(path, params):
    # return 0
    execution_command = [
        "/home/ubuntu/velox/build/velox/optimizer/benchmarkmul_test {}".format(params)
    ]
    result = float(
    subprocess.run(execution_command, stdout=subprocess.PIPE, shell=True).stdout
    )
    print("result", result)
    return result


if __name__ == "__main__":
    list_feature_size = [597540]
    list_sample_size = [1000]
    # list_sample_size = [100000]
    list_output_size = [1024]
    list_rewrite = ["false","true"]
    # list_mode = ["mul2joinAgg", "mul2joinAggHorizontal"]
    list_mode = ["mul2joinAggHorizontal"]
    list_velox_driver = [1,2,4,8]
    list_function_threads = [1,2,4,8]
    list_blocks = [4,8,16]
    list_split_disk = ["false","true"]
    num_repeat = 1
    list_batch_size = ["100", "1000", "10000", "100000"]

    run_configs = list(
        itertools.product(
            list_sample_size, list_feature_size, list_output_size, list_velox_driver, list_function_threads, list_blocks, list_batch_size, list_mode
        )
    )
    result_df = None
    for config in tqdm(run_configs):
        sample_size, feature_size, output_size, velox_driver, function_threads, list_blocks, batch_size, mode = config
        params = "-mode={} -feature_size={} -num_sample={} -num_repeat={} -rewrite={} -output_size={} -num_driver={} -num_function_threads={} -num_blocks={} -batch_size={} -split_disk={} -verbose=1".format(
            mode, feature_size, sample_size, num_repeat, "false", output_size, velox_driver, function_threads, list_blocks, batch_size, "true", 
        )
        print("velox driver:", velox_driver)
        print("function_threads:", function_threads)
        print("num_blocks:", list_blocks)
        print("batch_size:", batch_size)
        try:
            latency = run_cpp_program("/", params)
            df = pd.DataFrame(
                {
                    "mode": mode,
                    "feature_size": feature_size,
                    "num_sample": sample_size,
                    "output_size": output_size,
                    "num_driver": velox_driver,
                    "num_threads": function_threads,
                    "num_blocks": list_blocks,
                    "batch_size": batch_size,
                    "files_to_disk":"true",
                    "flag_rewrite":"false",
                    "latency": latency,
                },
                index=[0],
            )
        except Exception as e:
            df = pd.DataFrame(
                {
                    "mode": mode,
                    "feature_size": feature_size,
                    "num_sample": sample_size,
                    "output_size": output_size,
                    "num_driver": velox_driver,
                    "num_threads": function_threads,
                    "num_blocks": list_blocks,
                    "latency": e,
                },
                index=[0],
            )
        if result_df is None:
            result_df = df
        else:
            result_df = pd.concat([result_df, df], axis=0)

        result_df.to_csv("result_benchmark_arms_test.csv", index=False)

