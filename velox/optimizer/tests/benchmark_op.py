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
    list_l_max_index = [50, 100, 500, 1000, 2000, 5000, 10000, 50000]
    list_r_max_index = [50, 100, 500, 1000, 2000, 5000, 10000, 50000]
    list_dims1 = [100, 200, 500, 1000, 5000, 10000]
    list_dims2 = [100, 200, 500, 1000, 5000, 10000]
    list_reverse_order = ["false", "true"]
    list_op = ["HashJoin", "NestedLoopJoin"]
    # list_kernel = ["MatMul"]
    
    # list_feature_size = [8, 20]
    # list_batch_size = list(np.arange(20, 100, 20))
    # list_dims2 = [100]
    # mode = "benchmark_mul2joinagg"  # benchmark_mul2joinagg or benchmark_udf2torchdnn
    num_repeat = 10
    run_configs = list(
        itertools.product(
            list_op, list_l_max_index, list_r_max_index, list_dims1, list_dims2, list_reverse_order
        )
    )
    result_df = None
    for config in tqdm(run_configs):
        op, l_max_index, r_max_index, dim1, dim2, reverse_order = config
        params_base = "-mode=DB -op={} -l_max_index={} -r_max_index={} -dim1={} -dim2={} -reverse_order={}".format(
            op, l_max_index, r_max_index, dim1, dim2, reverse_order
        )
        try:
            print("[DEBUG] params: ", params_base)
            latency = run_cpp_program("/", params_base)
            df = pd.DataFrame(
                {
                    "op": op,
                    "l_max_index": l_max_index,
                    "r_max_index": r_max_index,
                    "dim1": dim1,
                    "dim2": dim2,
                    "reverse_order" :reverse_order,
                    "latency": latency,
                    "error": "",
                },
                index=[0],
            )
        except Exception as e:
            df = pd.DataFrame(
                {
                    "op": op,
                    "l_max_index": l_max_index,
                    "r_max_index": r_max_index,
                    "dim1": dim1,
                    "dim2": dim2,
                    "reverse_order" :reverse_order,
                    "latency": -1,
                    "error": e
                },
                index=[0],
            )
        if result_df is None:
            result_df = df
        else:
            result_df = pd.concat([result_df, df], axis=0)

        result_df.to_csv("result_benchmark_matmul.csv", index=False)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(result_df)
