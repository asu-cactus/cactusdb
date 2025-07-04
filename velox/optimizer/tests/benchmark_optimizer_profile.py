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
        "/home/velox/_build/release/velox/optimizer/tests/profile_query_generator {}".format(
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


def sample_model_structure(num_layer, model_scale):
    model_structure = []
    input_layer_range = None
    middle_layer_range = None
    output_layer_range = None
    if model_scale == "small":
        input_layer_range = (5, 100)
        middle_layer_range = (100, 256)
        output_layer_range = (1, 3)
    elif model_scale == "medium":
        input_layer_range = (500, 1000)
        middle_layer_range = (512, 1024)
        output_layer_range = (3, 10)
    elif model_scale == "large":
        input_layer_range = (1000, 10000)
        middle_layer_range = (512, 2048)
        output_layer_range = (10, 100)

    # at least one hidden layer
    assert num_layer >= 3

    for i in range(num_layer):
        if i == 0:
            num_node = random.randint(input_layer_range[0], input_layer_range[1])
        elif i == num_layer - 1:
            num_node = random.randint(output_layer_range[0], output_layer_range[1])
        else:
            num_node = random.randint(middle_layer_range[0], middle_layer_range[1])
        model_structure.append(num_node)
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


def configure_model_params(query_template, params_base, num_tag):
    if "user" in query_template:
        user_model_structure = sample_model_structure(
            sample_model_num_layer(), sample_model_scale("user")
        )
        num_user_dummy_features = user_model_structure[0]
        params_base = "{} -user_feature_size={}".format(
            params_base, num_user_dummy_features
        )
        write_model_structure_to_file(user_model_structure, "user")
    if "movie" in query_template:
        movie_model_structure = sample_model_structure(
            sample_model_num_layer(), sample_model_scale("movie")
        )
        num_movie_dummy_features = movie_model_structure[0]
        params_base = "{} -movie_feature_size={}".format(
            params_base, num_movie_dummy_features
        )
        write_model_structure_to_file(movie_model_structure, "movie")
    if "tag" in query_template:
        movie_tag_model_structure = sample_model_structure(
            sample_model_num_layer(), sample_model_scale("movie_tag")
        )
        movie_tag_model_structure[0] = num_tag
        params_base = "{} -num_tag={}".format(params_base, num_tag)
        write_model_structure_to_file(movie_tag_model_structure, "tag")
    return params_base


if __name__ == "__main__":
    os.environ["CD_PROFILE_W_FILTER"] = "True"
    # setting for synthetic data
    # list_num_user = [100, 500, 1000]
    # list_num_movie = [100, 500, 1000]
    list_num_user = np.arange(100, 8000, 100)
    list_num_user = np.random.choice(list_num_user, 20, replace=True)
    list_num_movie = np.arange(100, 8000, 100)
    list_num_movie = np.random.choice(list_num_movie, 20, replace=True)
    list_query_template = ["user", "movie", "movie_user", "movie_user_tag"]
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

    num_repeat = 1
    run_configs = list(
        itertools.product(
            list_query_template,
            list_num_user,
            list_num_movie,
            list_num_tag,
        )
    )

    # Generate permutations based on the query template
    effective_run_configs = []
    for query_template in list_query_template:
        if query_template == "user":
            for num_user in list_num_user:
                effective_run_configs.append((query_template, num_user, 1, 1))
        elif query_template == "movie":
            for num_movie in list_num_movie:
                effective_run_configs.append((query_template, 1, num_movie, 1))
        elif query_template == "movie_user":
            for num_user in list_num_user:
                for num_movie in list_num_movie:
                    effective_run_configs.append(
                        (query_template, num_user, num_movie, 1)
                    )
        elif query_template == "movie_user_tag":
            for num_user in list_num_user:
                for num_movie in list_num_movie:
                    for num_tag in list_num_tag:
                        effective_run_configs.append(
                            (query_template, num_user, num_movie, num_tag)
                        )

    run_configs = effective_run_configs

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

    result_df_name1 = "./generatedQueryPlan/result_optimizer_profile_v1_{}.csv".format(
        time_stamp
    )

    # if os.path.exists(result_df_name):
    #   new_result_df_name = "result_optimizer_profile_{}.csv".format(get_current_time(-3600))
    #   os.rename(result_df_name, new_result_df_name)

    for config in tqdm(run_configs):
        query_template, num_user, num_movie, num_tag = config

        params_base = "-query_template={} -verbose=1 -num_repeat={} -num_user={} -num_movie={} -num_tag={}".format(
            query_template, num_repeat, num_user, num_movie, num_tag
        )

        params_base = configure_model_params(query_template, params_base, num_tag)
        uuid_str = str(uuid.uuid4())
        serializedPlanPath = os.path.join(
            output_dir, "query", "{}.json".format(uuid_str)
        )
        tableStatsPath = os.path.join(output_dir, "stats", "{}.txt".format(uuid_str))

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
            tableStats = read_file("/home/velox/velox/optimizer/tests/tableStats.txt")

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
                    "num_user": num_user,
                    "num_movie": num_movie,
                    "num_tag": num_tag,
                    "serializedPlanPath": serializedPlanPath,
                    "tableStatsPath": tableStatsPath,
                    "executionTime": executionTime,
                    "params": params_base,
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
