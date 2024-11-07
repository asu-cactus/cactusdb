import subprocess
import pandas as pd
import numpy as np
import itertools
from tqdm.auto import tqdm


def run_cpp_program(path, params):
    # return 0
    execution_command = [
        "/home/velox/_build/release/velox/optimizer/tests/profile_query_generator {}".format(params)
    ]
    result = subprocess.run(execution_command, stdout=subprocess.PIPE, shell=True).stdout
    
    # print("result", result)
    return result

def read_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return content


if __name__ == "__main__":
    list_num_user = [100, 500, 1000, 5000]
    list_num_movie = [100, 500, 1000, 5000]
    list_num_tag = [25, 50, 100, 1000, 5000, 10000]

    num_repeat = 4
    run_configs = list(
        itertools.product(
            list_num_user, list_num_movie, list_num_tag, 
        )
    )
    result_df = None
    for config in tqdm(run_configs):
        num_user, num_movie, num_tag = config
        params_base = "-model=ml -verbose=1 -num_repeat={} -num_user={} -num_movie={} -num_tag={}".format(
            num_repeat, num_user, num_movie, num_tag
        )
        try:
            print("[DEBUG] params: ", params_base)
            latency = run_cpp_program("/", params_base)
            serializedPlan = read_file("/home/velox/velox/optimizer/tests/serializedQueryPlan.txt")
            # print(serializedPlan)
            executionTime = float(read_file("/home/velox/velox/optimizer/tests/executionLatency.txt"))
            # print(executionTime)
            df = pd.DataFrame(
                {
                    "num_user": num_user,
                    "num_movie": num_movie,
                    "num_tag": num_tag,
                    "serializedPlan": serializedPlan,
                    "executionTime": executionTime,
                    "error": "",
                },
                index=[0],
            )
        except Exception as e:
            df = pd.DataFrame(
                {
                    "num_user": num_user,
                    "num_movie": num_movie,
                    "num_tag": num_tag,
                    "serializedPlan": "",
                    "executionTime": "",
                    "error": e
                },
                index=[0],
            )
        if result_df is None:
            result_df = df
        else:
            result_df = pd.concat([result_df, df], axis=0)

        result_df.to_csv("result_optimizer_profile.csv", index=False, sep="|")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    # print(result_df)
