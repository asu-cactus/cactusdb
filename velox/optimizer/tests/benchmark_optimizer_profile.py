import subprocess
import pandas as pd
import numpy as np
import itertools
import uuid
import os
import time
import json
from tqdm.auto import tqdm

def get_current_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def remove_type_attribute(obj):
    if isinstance(obj, dict):
        # Use list() to avoid 'RuntimeError: dictionary changed size during iteration'
        for key in list(obj.keys()):
            if key == "type" or key == "outputType":
                del obj[key]
            elif key == "functionName" and obj[key] is None:
                del obj[key]
            else:
                # Recursively call the function on nested dictionaries
                remove_type_attribute(obj[key])
    elif isinstance(obj, list):
        # Recursively call the function on each element of the list
        for item in obj:
            remove_type_attribute(item)


if __name__ == "__main__":
    list_num_user = [100, 500, 1000]
    list_num_movie = [100, 500, 1000]
    list_num_tag = [25, 50, 100, 1000, 5000]
    list_query_template = ["user", "movie", "movie_user", "movie_user_tag"]

    num_repeat = 4
    run_configs = list(
        itertools.product(
            list_query_template, list_num_user, list_num_movie, list_num_tag, 
        )
    )
    result_df = None

    time_stamp = get_current_time()
    output_dir = os.path.join("generatedQueryPlan", time_stamp)
    create_path(output_dir)
    result_df_name = "result_optimizer_profile_{}.csv".format(time_stamp)

    for config in tqdm(run_configs):
        query_template, num_user, num_movie, num_tag = config
        params_base = "-query_template={} -verbose=1 -num_repeat={} -num_user={} -num_movie={} -num_tag={}".format(
            query_template, num_repeat, num_user, num_movie, num_tag
        )
        try:
            print("[DEBUG] params: ", params_base)
            latency = run_cpp_program("/", params_base)
            serializedPlan = read_file("/home/velox/velox/optimizer/tests/serializedQueryPlan.json")
            serializedPlan = json.loads(serializedPlan)
            remove_type_attribute(serializedPlan)
            serializedPlan = json.dumps(serializedPlan)
            uuid_str = str(uuid.uuid4())
            serializedPlanPath = os.path.join(output_dir, "{}.json".format(uuid_str))
            with open(serializedPlanPath, 'w') as file:
                file.write(serializedPlan)

            # print(serializedPlan)
            executionTime = float(read_file("/home/velox/velox/optimizer/tests/executionLatency.txt"))
            # print(executionTime)
            df = pd.DataFrame(
                {
                    "num_user": num_user,
                    "num_movie": num_movie,
                    "num_tag": num_tag,
                    "serializedPlanPath": serializedPlanPath,
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
                    "serializedPlanPath": serializedPlanPath,
                    "executionTime": "",
                    "error": e
                },
                index=[0],
            )
        if result_df is None:
            result_df = df
        else:
            result_df = pd.concat([result_df, df], axis=0)

        result_df.to_csv(result_df_name, index=False, sep="|")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    # print(result_df)
