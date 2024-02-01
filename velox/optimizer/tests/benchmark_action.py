import subprocess
import pandas as pd
import numpy as np
import itertools
from tqdm.auto import tqdm


def run_cpp_program(path, params):
    # return 0
    execution_command = [
        "/home/velox/_build/release/velox/optimizer/tests/mcts_test {}".format(params)
    ]
    result = float(
        subprocess.run(execution_command, stdout=subprocess.PIPE, shell=True).stdout
    )
    print("result", result)
    return result


if __name__ == "__main__":
    list_feature_size = [500, 1000, 2000, 5000, 10000, 50000, 100000]
    list_sample_size = [500, 1000]
    list_transform = ["false", "true"]
    mode = "benchmark_mul2joinagg"  # benchmark_mul2joinagg or benchmark_udf2torchdnn
    num_repeat = 5
    run_configs = list(
        itertools.product(list_feature_size, list_sample_size, list_transform)
    )
    result_df = None
    for config in tqdm(run_configs):
        feature_size, sample_size, transform_flag = config
        params = "--mode={} --feature_size={} --num_sample={} --num_repeat={} --rewrite={}".format(
            mode, feature_size, sample_size, num_repeat, transform_flag
        )
        latency = run_cpp_program("/", params)
        df = pd.DataFrame(
            {
                "feature_size": feature_size,
                "num_sample": sample_size,
                "rewrite": transform_flag,
                "latency": latency,
            },
            index=[0],
        )
        if result_df is None:
            result_df = df
        else:
            result_df = pd.concat([result_df, df], axis=0)

        result_df.to_csv("result_{}.csv".format(mode), index=False)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(result_df)
