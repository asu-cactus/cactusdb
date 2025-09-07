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
        "{} {}".format(
           path, params
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


def sample_model_kernel_by_layer(
    hidden_size, model_kernel_names, model_kernel_sizes, is_last_layer
):
    # add matrix multiplication
    model_kernel_names.append("MatMul")
    model_kernel_sizes.append(hidden_size)
    # add matrix addition
    model_kernel_names.append("MatAdd")
    model_kernel_sizes.append(hidden_size)
    if not is_last_layer:
        # sample activation function
        activation_functions = ["ReLU", "Sigmoid"]
        model_kernel_names.append(random.choice(activation_functions))
        model_kernel_sizes.append(hidden_size)
    else:
        last_layer_functions = ["Sigmoid", "Softmax", "Softmax+Argmax"]
        last_layer_kernel = random.choice(last_layer_functions)
        if last_layer_kernel == "Softmax+Argmax":
            model_kernel_names.append("Softmax")
            model_kernel_sizes.append(hidden_size)
            model_kernel_names.append("Argmax")
            model_kernel_sizes.append(hidden_size)
        elif last_layer_kernel == "Softmax":
            model_kernel_names.append("Softmax")
            model_kernel_sizes.append(hidden_size)
        elif last_layer_kernel == "Sigmoid":
            model_kernel_names.append("Sigmoid")
            model_kernel_sizes.append(hidden_size)

    return


def write_str_to_file(str_content, filename):
    with open(filename, "w") as file:
        file.write(str_content)


if __name__ == "__main__":
    # setting for benchmark
    # list_num_data = [10]
    num_data = 10000

    time_stamp = get_current_time()
    output_dir = os.path.join("generatedQueryPlan", time_stamp)
    create_path(output_dir)
    create_path(os.path.join(output_dir, "query"))

    result_df_name1 = "./generatedQueryPlan/result_optimizer_profile_{}.csv".format(
        time_stamp
    )

    num_repeat = 4
    sampled_model_kernel_name_path = (
        "/home/velox/velox/optimizer/tests/_sampledModel/model_kernel_name.txt"
    )
    sampled_model_kernel_size_path = (
        "/home/velox/velox/optimizer/tests/_sampledModel/model_kernel_size.txt"
    )

    result_df = None

    for i in tqdm(range(num_data)):
        # sample feature size
        num_feature = np.random.randint(10, 5000)
        # sample number of layers, minimun is only output layer
        num_layer = np.random.randint(1, 6)
        model_kernel_names = []
        model_kernel_sizes = []
        for j in range(num_layer):
            hidden_size = np.random.randint(10, 2000)
            is_last_layer = j == num_layer - 1
            sample_model_kernel_by_layer(
                hidden_size, model_kernel_names, model_kernel_sizes, is_last_layer
            )

        write_str_to_file(
            " ".join(map(str, model_kernel_names)), sampled_model_kernel_name_path
        )
        write_str_to_file(
            " ".join(map(str, model_kernel_sizes)), sampled_model_kernel_size_path
        )

        rewrite = random.choice(['true', 'false'])

        params_base = "-modelType={} -verbose=1 -num_repeat={} -feature_size={} -num_data={} -rewrite={}".format(
            "ffnn", num_repeat, num_feature, num_data, rewrite
        )

        uuid_str = str(uuid.uuid4())
        serializedPlanPath = os.path.join(
            output_dir, "query", "{}.json".format(uuid_str)
        )

        try:
            latency = run_cpp_program("/home/velox/_build/release/velox/optimizer/tests/model_query_profiler", params_base)
            serializedPlan = read_file(
                "/home/velox/velox/optimizer/tests/serializedQueryPlan.json"
            )
            serializedPlan = json.loads(serializedPlan)
            remove_type_attribute(serializedPlan)
            serializedPlan = json.dumps(serializedPlan)

            with open(serializedPlanPath, "w") as file:
                file.write(serializedPlan)


            # read execution time
            executionTime = float(
                read_file("/home/velox/velox/optimizer/tests/executionLatency.txt")
            )
            os.remove("/home/velox/velox/optimizer/tests/executionLatency.txt")

            # print(executionTime)
            df = pd.DataFrame(
                {
                    "num_data": num_data,
                    "num_feature": num_feature,
                    "kernel_name": ",".join(map(str, model_kernel_names)),
                    "kernel_size": ",".join(map(str, model_kernel_sizes)),
                    "serializedPlanPath": serializedPlanPath,
                    "executionTime": executionTime,
                    "params": params_base,
                    "error": "",
                },
                index=[0],
            )
        except Exception as e:
            df = pd.DataFrame(
                {
                    "num_data": num_data,
                    "num_feature": num_feature,
                    "kernel_name": ",".join(map(str, model_kernel_names)),
                    "kernel_size": ",".join(map(str, model_kernel_sizes)),
                    "serializedPlanPath": serializedPlanPath,
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

    # num_repeat = 1
    # run_configs = list(
    #     itertools.product(
    #         list_query_template,
    #         list_num_user,
    #         list_num_movie,
    #         list_num_tag,
    #     )
    # )

    # Generate permutations based on the query template
    # effective_run_configs = []
    # for query_template in list_query_template:
    #     if query_template == "user":
    #         for num_user in list_num_user:
    #             effective_run_configs.append((query_template, num_user, 1, 1))
    #     elif query_template == "movie":
    #         for num_movie in list_num_movie:
    #             effective_run_configs.append((query_template, 1, num_movie, 1))
    #     elif query_template == "movie_user":
    #         for num_user in list_num_user:
    #             for num_movie in list_num_movie:
    #                 effective_run_configs.append(
    #                     (query_template, num_user, num_movie, 1)
    #                 )
    #     elif query_template == "movie_user_tag":
    #         for num_user in list_num_user:
    #             for num_movie in list_num_movie:
    #                 for num_tag in list_num_tag:
    #                     effective_run_configs.append(
    #                         (query_template, num_user, num_movie, num_tag)
    #                     )

    # run_configs = effective_run_configs

    # # random.shuffle(run_configs)
    # result_df = None

    # # TODO: clean up for development
    # # if os.path.exists("./generatedQueryPlan"):
    # #   shutil.rmtree("./generatedQueryPlan")
    # time_stamp = get_current_time()
    # output_dir = os.path.join("generatedQueryPlan", time_stamp)
    # create_path(output_dir)
    # create_path(os.path.join(output_dir, "query"))
    # create_path(os.path.join(output_dir, "stats"))

    # # TODO: use time_stamp to name the result file after finalizing the code

    # result_df_name1 = "./generatedQueryPlan/result_optimizer_profile_{}.csv".format(
    #     time_stamp
    # )
    # result_df_name2 = "result_optimizer_profile.csv"

    # # if os.path.exists(result_df_name):
    # #   new_result_df_name = "result_optimizer_profile_{}.csv".format(get_current_time(-3600))
    # #   os.rename(result_df_name, new_result_df_name)

    # for config in tqdm(run_configs):
    #     query_template, num_user, num_movie, num_tag = config

    #     params_base = "-query_template={} -verbose=1 -num_repeat={} -num_user={} -num_movie={} -num_tag={}".format(
    #         query_template, num_repeat, num_user, num_movie, num_tag
    #     )

    #     params_base = configure_model_params(query_template, params_base, num_tag)
    #     uuid_str = str(uuid.uuid4())
    #     serializedPlanPath = os.path.join(
    #         output_dir, "query", "{}.json".format(uuid_str)
    #     )
    #     tableStatsPath = os.path.join(output_dir, "stats", "{}.txt".format(uuid_str))

    #     try:
    #         print("[DEBUG] params: ", params_base)
    #         latency = run_cpp_program("/", params_base)

    #         # generate uuid for the current one

    #         # process serialized plan
    #         serializedPlan = read_file(
    #             "/home/velox/velox/optimizer/tests/serializedQueryPlan.json"
    #         )
    #         serializedPlan = json.loads(serializedPlan)
    #         remove_type_attribute(serializedPlan)
    #         serializedPlan = json.dumps(serializedPlan)

    #         with open(serializedPlanPath, "w") as file:
    #             file.write(serializedPlan)

    #         # process query table statistics
    #         tableStats = read_file("/home/velox/velox/optimizer/tests/tableStats.txt")

    #         with open(tableStatsPath, "w") as file:
    #             file.write(tableStats)

    #         # read execution time
    #         executionTime = float(
    #             read_file("/home/velox/velox/optimizer/tests/executionLatency.txt")
    #         )
    #         os.remove("/home/velox/velox/optimizer/tests/executionLatency.txt")

    #         # print(executionTime)
    #         df = pd.DataFrame(
    #             {
    #                 "num_user": num_user,
    #                 "num_movie": num_movie,
    #                 "num_tag": num_tag,
    #                 "serializedPlanPath": serializedPlanPath,
    #                 "tableStatsPath": tableStatsPath,
    #                 "executionTime": executionTime,
    #                 "params": params_base,
    #                 "error": "",
    #             },
    #             index=[0],
    #         )
    #     except Exception as e:
    #         df = pd.DataFrame(
    #             {
    #                 "num_user": num_user,
    #                 "num_movie": num_movie,
    #                 "num_tag": num_tag,
    #                 "serializedPlanPath": serializedPlanPath,
    #                 "tableStatsPath": tableStatsPath,
    #                 "executionTime": "",
    #                 "params": params_base,
    #                 "error": e,
    #             },
    #             index=[0],
    #         )
    #     if result_df is None:
    #         result_df = df
    #     else:
    #         result_df = pd.concat([result_df, df], axis=0)

    #     result_df.to_csv(result_df_name1, index=False, sep="|")
    #     result_df.to_csv(result_df_name2, index=False, sep="|")

    # pd.set_option("display.max_rows", None)
    # pd.set_option("display.max_columns", None)
    # print(result_df)
