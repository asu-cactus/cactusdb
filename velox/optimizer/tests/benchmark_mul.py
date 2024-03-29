import subprocess
import pandas as pd
import numpy as np
import itertools
from tqdm.auto import tqdm


def run_cpp_program(path, params):
    # return 0
    execution_command = [
        "/home/local/ASUAD/qlin36/velox/build/velox/optimizer/benchmarkmul_oom_test {}".format(params)
    ]
    result = float(
    subprocess.run(execution_command, stdout=subprocess.PIPE, shell=True).stdout
    )
    print("result", result)
    return result


if __name__ == "__main__":
    list_feature_size = [100000]
    list_sample_size = [10000]
    # list_sample_size = [100000]
    list_output_size = [10240]
    # list_rewrite = ["false","true"]
    list_rewrite = ["false"]
    # list_mode = ["mul2joinAgg", "mul2joinAggHorizontal"]
    list_mode = ["mul2joinAggHorizontal"]
    list_velox_driver = [8]
    list_function_threads = [1]
    list_blocks = [8]
    list_split_disk = ["false"]
    list_option =[1]
    num_repeat = 1
    # list_batch_size = ["100", "1000", "10000", "100000"]
    list_batch_size = ["10000"] # kPreferredOutputBatchBytes 2500000 (597540*4) bytes per item, until 10000 *2500000, then kMaxOutputBatchRows determine the output, we all set it as 250000000000 to make kMaxOutputBatchRows control it.

    run_configs = list(
        itertools.product(
            list_sample_size, list_feature_size, list_output_size, list_velox_driver, list_function_threads, list_blocks, list_batch_size, list_mode, list_rewrite, list_split_disk, list_option
        )
    )
    result_df = None
    for config in tqdm(run_configs):
        sample_size, feature_size, output_size, velox_driver, function_threads, list_blocks, batch_size, mode, rewrite, disk, option = config
        params = "-mode={} -feature_size={} -num_sample={} -num_repeat={} -rewrite={} -output_size={} -num_driver={} -num_function_threads={} -num_blocks={} -batch_size={} -split_disk={} -option={} -verbose=1".format(
            mode, feature_size, sample_size, num_repeat, rewrite, output_size, velox_driver, function_threads, list_blocks, batch_size, disk, option 
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
                    "files_to_disk":disk,
                    "flag_rewrite":rewrite,
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
                    "batch_size": batch_size,
                    "files_to_disk":disk,
                    "flag_rewrite":rewrite,
                    "latency": e,
                },
                index=[0],
            )
        if result_df is None:
            result_df = df
        else:
            result_df = pd.concat([result_df, df], axis=0)

        result_df.to_csv("result_benchmark_arms_11_nodisk_h_oom_test.csv", index=False)

