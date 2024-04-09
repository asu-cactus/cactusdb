import subprocess
import pandas as pd
import numpy as np
import itertools
from tqdm.auto import tqdm


def run_cpp_program(path, params):
    # return 0
    execution_command = [
        "/home/velox/_build/release/velox/ml_functions/ml_kernel_benchmark {}".format(params)
    ]
    result = float(
        subprocess.run(execution_command, stdout=subprocess.PIPE, shell=True).stdout
    )
    print("result", result)
    return result


if __name__ == "__main__":
    list_feature_size = [8, 20, 40, 80, 100, 200, 500, 1000, 2000, 5000]
    list_batch_size = []
    list_batch_size += list(np.arange(20, 100, 20))
    list_batch_size += list(np.arange(100, 1000, 100))
    list_batch_size += list(np.arange(1000, 10000, 500))
    # list_transform = ["false", "true"]
    list_model = ["FFNN"]
    # list_kernel = ["MatMul"]
    list_l1_size = [100, 200, 500, 1000, 5000]
    list_l2_size = [100, 200, 500, 1000, 5000]
    # list_feature_size = [8, 20]
    # list_batch_size = list(np.arange(20, 100, 20))
    # list_dims2 = [100]
    # mode = "benchmark_mul2joinagg"  # benchmark_mul2joinagg or benchmark_udf2torchdnn
    num_repeat = 20
    run_configs = list(
        itertools.product(
            list_model, list_batch_size, list_feature_size, list_l1_size, list_l2_size
        )
    )
    result_df = None
    for config in tqdm(run_configs):
        model, batch_size, feature_size, l1_size, l2_size = config
        params_base = "-mode=TorchNN -model={} -batch_size={} -feature_size={} -num_repeat={} -l1size={} -l2size={}".format(
            model, batch_size, feature_size, num_repeat, l1_size, l2_size
        )
        try:
            print("[DEBUG] params: ", params_base)
            latency = run_cpp_program("/", params_base)
            df = pd.DataFrame(
                {
                    "model": model,
                    "batch_size": batch_size,
                    "feature_size": feature_size,
                    "l1size": l1_size,
                    "l2size": l2_size,
                    "latency": latency,
                    "error": "",
                },
                index=[0],
            )
        except Exception as e:
            df = pd.DataFrame(
                {
                    "model": model,
                    "batch_size": batch_size,
                    "feature_size": feature_size,
                    "l1size": l1_size,
                    "l2size": l2_size,
                    "latency": latency,
                    "error": e
                },
                index=[0],
            )
        if result_df is None:
            result_df = df
        else:
            result_df = pd.concat([result_df, df], axis=0)

        result_df.to_csv("result_benchmark_torchnn.csv", index=False)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(result_df)
