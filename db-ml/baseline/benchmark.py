import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pipeline
import pandas as pd
import warnings
import load_data_to_db
import itertools
import datetime

warnings.filterwarnings("ignore")
from tqdm.auto import tqdm


def benchmark_two_tower_model_pipeline_pytorch(num_sample=500, num_loop=10, **kwargs):
    two_tower_model_pipeline = pipeline.TwoTowerModelPipelinePyTorch(
        num_sample=num_sample, num_loop=num_loop
    )
    benchmark_result = two_tower_model_pipeline.run_pipeline()
    return benchmark_result


def benchmark_two_tower_model_pipeline_tf(num_sample=500, num_loop=10, **kwargs):
    two_tower_model_pipeline = pipeline.TwoTowerModelPipelineTF(
        num_sample=num_sample, num_loop=num_loop
    )
    benchmark_result = two_tower_model_pipeline.run_pipeline()
    return benchmark_result


def benchmark_two_tower_model_pipeline_evadb(num_sample=500, num_loop=10, **kwargs):
    two_tower_model_pipeline = pipeline.TwoTowerModelPipelineEvaDB(
        num_sample=num_sample, num_loop=num_loop
    )
    benchmark_result = two_tower_model_pipeline.run_pipeline()
    return benchmark_result


def benchmark_ffnn_pipeline_tf(
    list_hidden_layer_sizes,
    num_sample=500,
    num_total_record=1000,
    num_loop=10,
    **kwargs
):
    ffnn_pipeline = pipeline.FFNNPipelineTF(
        list_hidden_layer_sizes=list_hidden_layer_sizes,
        num_sample=num_sample,
        num_total_record=num_total_record,
        num_loop=num_loop,
        ffnn_table_name=kwargs['ffnn_table_name']
    )
    benchmark_result = ffnn_pipeline.run_pipeline()
    return benchmark_result


def benchmark_ffnn_pipeline_pytorch(
    list_hidden_layer_sizes,
    num_sample=500,
    num_total_record=1000,
    num_loop=10,
    **kwargs
):
    ffnn_pipeline = pipeline.FFNNPipelinePyTorch(
        list_hidden_layer_sizes=list_hidden_layer_sizes,
        num_sample=num_sample,
        num_total_record=num_total_record,
        num_loop=num_loop,
        ffnn_table_name=kwargs['ffnn_table_name']
    )
    benchmark_result = ffnn_pipeline.run_pipeline()
    return benchmark_result


def benchmark_ffnn_pipeline_evadb(
    list_hidden_layer_sizes,
    num_sample=500,
    num_total_record=1000,
    num_loop=10,
    **kwargs
):
    ffnn_pipeline = pipeline.FFNNPipelineEvaDB(
        list_hidden_layer_sizes=list_hidden_layer_sizes,
        num_sample=num_sample,
        num_total_record=num_total_record,
        num_loop=num_loop,
        ffnn_table_name=kwargs['ffnn_table_name']
    )
    benchmark_result = ffnn_pipeline.run_pipeline()
    return benchmark_result

def benchmark_ffnn_pipeline_sparksql(
    list_hidden_layer_sizes,
    num_sample=500,
    num_total_record=1000,
    num_loop=10,
    **kwargs
):
    ffnn_pipeline = pipeline.FFNNPipelineSparkSQL(
        list_hidden_layer_sizes=list_hidden_layer_sizes,
        num_sample=num_sample,
        num_total_record=num_total_record,
        num_loop=num_loop,
        ffnn_table_name=kwargs['ffnn_table_name']
    )
    benchmark_result = ffnn_pipeline.run_pipeline()
    return benchmark_result

def benchmark_ffnn_pipeline_sparksqlhadoop(
    list_hidden_layer_sizes,
    num_sample=500,
    num_total_record=1000,
    num_loop=10,
    **kwargs
):
    ffnn_pipeline = pipeline.FFNNPipelineSparkSQLHadoop(
        list_hidden_layer_sizes=list_hidden_layer_sizes,
        num_sample=num_sample,
        num_total_record=num_total_record,
        num_loop=num_loop,
        ffnn_table_name=kwargs['ffnn_table_name']
    )
    benchmark_result = ffnn_pipeline.run_pipeline()
    return benchmark_result


def benchmark_twotower_model():
    list_benchmark = []
    list_benchmark += [benchmark_two_tower_model_pipeline_pytorch]
    list_benchmark += [benchmark_two_tower_model_pipeline_tf]
    list_benchmark += [benchmark_two_tower_model_pipeline_evadb]
    list_num_sample = [5000]
    result_df = None
    num_loop = 5
    today = datetime.date.today()
    formatted_date = today.strftime("%m-%d-%Y")

    result_output_file = "result_{}_{}.csv".format("twotower", formatted_date)
    list_run_config = list(itertools.product(list_num_sample, list_benchmark))
    print("[INFO] benchmark config to run: \n \t {}".format(list_run_config))

    for num_sample, benchmark in tqdm(list_run_config):
        result = benchmark(
            num_sample=num_sample,
            num_loop=num_loop,
        )
        if result_df is None:
            result_df = result
            result.to_csv(result_output_file, sep=",", index=False, mode="a")
        else:
            result.to_csv(result_output_file, sep=",", index=False, mode="a", header=False)

def benchmark_ffnn():
    list_benchmark = []
    # list_benchmark += [benchmark_ffnn_pipeline_sparksql]
    # list_benchmark += [benchmark_ffnn_pipeline_evadb]
    # list_benchmark += [benchmark_ffnn_pipeline_tf]
    # list_benchmark += [benchmark_ffnn_pipeline_pytorch]
    # list_benchmark += [benchmark_ffnn_pipeline_sparksqlhadoop]
    num_feature = 100
    list_hidden_layer_sizes = [num_feature, 1024, 14588]
    # list_num_sample = [100, 500, 1000, 5000]
    list_num_sample = [6000]
    result_df = None
    num_loop = 5
    num_total_record = 100000
    
    today = datetime.date.today()
    formatted_date = today.strftime("%m-%d-%Y")

    result_output_file = "result_{}_{}.csv".format(num_feature, formatted_date)
    
    ffnn_table_name = "ffnn_data_{}".format(num_feature)
    load_data_to_db.load_ffnn_data_to_postgres(num_total_record, num_feature, ffnn_table_name)
    list_run_config = list(itertools.product(list_num_sample, list_benchmark))
    print("[INFO] benchmark config to run: \n \t {}".format(list_run_config))

    for num_sample, benchmark in tqdm(list_run_config):
        result = benchmark(
            num_sample=num_sample,
            num_loop=num_loop,
            list_hidden_layer_sizes=list_hidden_layer_sizes,
            num_total_record=num_total_record,
            ffnn_table_name = ffnn_table_name
        )
        result['nn'] = str(list_hidden_layer_sizes)
        if result_df is None:
            result_df = result
            result.to_csv(result_output_file, sep=",", index=False, mode="a")
        else:
            result.to_csv(result_output_file, sep=",", index=False, mode="a", header=False)


if __name__ == "__main__":
    benchmark_twotower_model()
    # benchmark_ffnn()
