import subprocess
import pandas as pd
import random
import numpy as np
import itertools
import uuid
import os
import time
import json
import shutil
from tqdm.auto import tqdm
import pdb


def get_current_time(timeshift=0):
    current_time = time.localtime(time.time() + timeshift)
    return time.strftime("%Y-%m-%d-%H-%M-%S", current_time)


def create_path(path, overwrite=False):
    if os.path.exists(path) and overwrite:
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


def run_cpp_program(path, params):
    # return 0
    execution_command = [
        "../../../_build/release/velox/optimizer/tests/profile_query_generator {}".format(
            params
        )
    ]
    result = subprocess.run(
        execution_command,
        stdout=subprocess.PIPE,
        shell=True,
        timeout=60 * 15,  # 15 minutes
    ).stdout

    # print("result", result)
    return result


def collect_movielens_stats():
    execution_command = [
        "/home/velox/_build/release/velox/optimizer/tests/reusable_mcts_test -mode=collect_ml_stats"
    ]
    result = subprocess.run(
        execution_command,
        stdout=subprocess.PIPE,
        shell=True,
        timeout=60 * 30,  # 30 minutes
    ).stdout

    # print("result", result)
    return result


def read_file(filename):
    with open(filename, "r") as file:
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


def sample_model_structure(num_layer, model_scale, input_size, output_size):
    # at least one hidden layer
    assert num_layer >= 3
    num_hidden_layers = num_layer - 2  # excluding input and output layers
    model_structure = [input_size]

    if model_scale == "small":
        layer_range = (100, 256)
    elif model_scale == "medium":
        layer_range = (512, 1024)
    elif model_scale == "large":
        layer_range = (512, 2048)
    else:
        raise ValueError("Invalid model_scale: ", model_scale)

    for _ in range(num_hidden_layers):
        num_node = random.randint(layer_range[0], layer_range[1])
        model_structure.append(num_node)
    model_structure.append(output_size)
    return model_structure


def sample_model_num_layer():
    return 4 if random.random() < 0.6 else 3


def sample_model_scale(query_template):
    if query_template == "user":
        return "small" if random.random() < 0.6 else "medium"
    elif query_template == "movie":
        return "medium" if random.random() < 0.6 else "large"
    elif query_template == "movie_tag":
        rand_val = random.random()
        if rand_val < 0.2:
            return "small"
        elif rand_val < 0.5:
            return "medium"
        else:
            return "large"
    else:
        raise ValueError("Invalid query_template: ", query_template)


def write_model_structure_to_file(model_structure, table):
    if table == "user":
        path = "/home/velox/velox/optimizer/tests/user_dummy_model_structure.txt"
    elif table == "movie":
        path = "/home/velox/velox/optimizer/tests/movie_dummy_model_structure.txt"
    elif table == "tag":
        path = "/home/velox/velox/optimizer/tests/tag_dummy_model_structure.txt"
    else:
        raise ValueError("Invalid table: ", table)
    with open(path, "w") as file:
        file.write(" ".join(map(str, model_structure)))


def configure_model_params(query_template, input_size, output_size):
    if "template" in query_template:
        user_model_structure = sample_model_structure(
            sample_model_num_layer(),
            sample_model_scale("user"),
            input_size,
            output_size,
        )
        write_model_structure_to_file(user_model_structure, "user")
    else:
        raise ValueError("Invalid query_template: ", query_template)


if __name__ == "__main__":
    os.environ["CD_PROFILE_W_FILTER"] = "True"
    # setting for synthetic data
    # list_num_user = [100, 500, 1000]
    # list_num_movie = [100, 500, 1000]
    list_num_user = np.arange(100, 8000, 100)
    list_num_user = np.random.choice(list_num_user, 20, replace=True)
    list_num_movie = np.arange(100, 8000, 100)
    list_num_movie = np.random.choice(list_num_movie, 20, replace=True)
    # list_query_template = ["user", "movie", "movie_user", "movie_user_tag"]
    list_query_template = ["template5"]
    list_num_tag = np.arange(50, 5000, 50)
    list_num_tag = np.random.choice(list_num_tag, 5, replace=True)
    # list_num_user = [100]
    # list_num_movie = [50]
    # list_num_tag = [25]

    # setting for movielens dataset
    # list_num_user = np.arange(100, 2000, 100)
    # list_num_user = np.random.choice(list_num_user, 10, replace=True)
    # list_num_movie = np.arange(100, 2000, 100)
    # list_num_movie = np.random.choice(list_num_movie, 1, replace=True)
    # list_query_template = ["ml-q1", "ml-q2", "ml-q3"]
    # list_num_tag = np.arange(50, 5000, 50)
    # list_num_tag = np.random.choice(list_num_tag, 5, replace=True)

    if "ml-q1" in list_query_template:
        collect_movielens_stats()

    # random.shuffle(run_configs)
    result_df = None

    # TODO: clean up for development
    # if os.path.exists("./generatedQueryPlan"):
    #   shutil.rmtree("./generatedQueryPlan")
    time_stamp = get_current_time()
    output_dir = os.path.join("generatedQueryPlan", time_stamp)
    create_path(output_dir)
    create_path(os.path.join(output_dir, "query"))
    create_path(os.path.join(output_dir, "stats"))

    # TODO: use time_stamp to name the result file after finalizing the code

    result_df_name1 = "./generatedQueryPlan/query_benchmark_results_{}.csv".format(
        time_stamp
    )

    # if os.path.exists(result_df_name):
    #   new_result_df_name = "result_optimizer_profile_{}.csv".format(get_current_time(-3600))
    #   os.rename(result_df_name, new_result_df_name)
    # query_template = "template7"  # hard code for now, can be changed later
    list_query_templates = ["template" + str(i) for i in range(4, 11)]
    input_size = 3
    output_size = 3706
    for _ in tqdm(range(10000)):
        for query_template in list_query_templates:
            params_base = f"-query_template={query_template} -workload=movielens -verbose=1 -num_repeat=1 -rewrite=true"

            # Hard code num_tag = 1
            configure_model_params(query_template, input_size, output_size)

            uuid_str = str(uuid.uuid4())
            serializedPlanPath = os.path.join(
                output_dir, "query", "{}.json".format(uuid_str)
            )
            tableStatsPath = os.path.join(
                output_dir, "stats", "{}.txt".format(uuid_str)
            )

            try:
                print("[DEBUG] params: ", params_base)
                latency = run_cpp_program("/", params_base)

                # generate uuid for the current one

                # process serialized plan
                serializedPlan = read_file(
                    "/home/velox/velox/optimizer/tests/serializedQueryPlan.json"
                )
                serializedPlan = json.loads(serializedPlan)
                remove_type_attribute(serializedPlan)
                serializedPlan = json.dumps(serializedPlan)

                with open(serializedPlanPath, "w") as file:
                    file.write(serializedPlan)

                # process query table statistics
                tableStats = read_file(
                    "/home/velox/velox/optimizer/tests/tableStats_movielens.txt"
                )

                with open(tableStatsPath, "w") as file:
                    file.write(tableStats)

                # read execution time
                executionTime = float(
                    read_file("/home/velox/velox/optimizer/tests/executionLatency.txt")
                )
                os.remove("/home/velox/velox/optimizer/tests/executionLatency.txt")

                # print(executionTime)
                df = pd.DataFrame(
                    {
                        "num_user": -1,
                        "num_movie": 1,
                        "num_tag": 1,
                        "workload": "movielens",
                        "template": query_template,
                        "serializedPlanPath": serializedPlanPath,
                        "tableStatsPath": tableStatsPath,
                        "executionTime": executionTime,
                        "params": params_base,
                        "error": "",
                    },
                    index=[0],
                )
            except Exception as e:
                print("Error occurred: ", e)
                df = pd.DataFrame(
                    {
                        "num_user": 1,
                        "num_movie": 1,
                        "num_tag": 1,
                        "workload": "movielens",
                        "template": query_template,
                        "serializedPlanPath": serializedPlanPath,
                        "tableStatsPath": tableStatsPath,
                        "executionTime": "",
                        "params": params_base,
                        "error": e,
                    },
                    index=[0],
                )
            if result_df is None:
                result_df = df
            else:
                result_df = pd.concat([result_df, df], axis=0)

            result_df.to_csv(result_df_name1, index=False, sep="|")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    # print(result_df)
