import subprocess
import pandas as pd
import numpy as np
import itertools
from tqdm.auto import tqdm


def run_cpp_program(path, params):
    # return 0
    execution_command = [
        "/home/ubuntu/velox/build/velox/optimizer/cost4Configs_test {}".format(params)
    ]
    result = float(
    subprocess.run(execution_command, stdout=subprocess.PIPE, shell=True).stdout
    )
    print("result", result)
    return result


if __name__ == "__main__":
    # list_feature_size = [1,10,100,1000,10000, 100000]
    list_feature_size = [1024]
    # list_sample_size = [100000, 1000000, 2000000,3000000]
    list_sample_size = [100,1000,10000,100000]
    # list_sample_size = [100000]
    list_output_size = [1024]

    list_velox_driver = [1,2,3,4,5,6,7,8]
    list_eigen_threads = [1,2,3,4,5,6,7,8]
    list_torch_threads = [1,2,3,4,5,6,7,8]
    # list_blocks = [4,8,16,32,64,128,256,512,1024]
    list_blocks = [8]
    list_option =[1]
    num_repeat = 1
    # list_batch_size = ["100", "1000", "10000", "100000"]
    list_batch_size = ["100", "1000", "10000", "100000"] # kPreferredOutputBatchBytes 2500000 (597540*4) bytes per item, until 10000 *2500000, then kMaxOutputBatchRows determine the output, we all set it as 250000000000 to make kMaxOutputBatchRows control it.

    run_configs = list(
        itertools.product(
            list_sample_size, list_feature_size, list_output_size, list_velox_driver, list_eigen_threads, list_torch_threads, list_blocks, list_batch_size, list_option
        )
    )
    result_df = None
    for config in tqdm(run_configs):
        sample_size, feature_size, output_size, velox_driver, eigen_threads, torch_threads, list_blocks, batch_size, option = config
        params = "-feature_size={} -num_sample={} -num_repeat={} -output_size={} -num_driver={} -num_eigen={} -num_torch={} -num_blocks={} -batch_size={}  -option={} -verbose=1".format(
            feature_size, sample_size, num_repeat, output_size, velox_driver, eigen_threads, torch_threads, list_blocks, batch_size, option 
        )
        print("velox driver:", velox_driver)
        print("eigen_threads:", eigen_threads)
        print("torch_threads:", torch_threads)
        print("feature_sizes:", feature_size)
        print("num_sample:", sample_size)

        try:
            latency = run_cpp_program("/", params)
            df = pd.DataFrame(
                {
                    "feature_size": feature_size,
                    "num_sample": sample_size,
                    "output_size": output_size,
                    "num_driver": velox_driver,
                    "eigen_threads": eigen_threads,
                    "torch_threads": torch_threads,
                    # "num_blocks": list_blocks,
                    "batch_size": batch_size,
                    "latency": latency,
                },
                index=[0],
            )
        except Exception as e:
            df = pd.DataFrame(
                {
                    "feature_size": feature_size,
                    "num_sample": sample_size,
                    "output_size": output_size,
                    "num_driver": velox_driver,
                    "eigen_threads": eigen_threads,
                    "torch_threads": torch_threads,
                    # "num_blocks": list_blocks,
                    "batch_size": batch_size,
                    "latency": e,
                },
                index=[0],
            )
        if result_df is None:
            result_df = df
        else:
            result_df = pd.concat([result_df, df], axis=0)

        result_df.to_csv("FFNN_threads_cost_test2.csv", index=False)