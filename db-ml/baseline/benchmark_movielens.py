import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS']='ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
tf.get_logger().setLevel('ERROR')
import pipeline
import pandas as pd
import warnings
import load_data_to_db
import itertools
import datetime

from tqdm.auto import tqdm

def benchmark_movielens_q1_dl(num_loop=10, **kwargs):
    movielens_q1_pipeline = pipeline.MovielensQ1PipelineDLCentric(
        num_loop=num_loop
    )
    benchmark_result = movielens_q1_pipeline.run_pipeline()
    return benchmark_result

def benchmark_movielens_q1_evadb(num_loop=10, **kwargs):
    movielens_q1_pipeline_evadb = pipeline.MovielensQ1PipelineEvaDB(
        num_loop=num_loop
    )
    benchmark_result = movielens_q1_pipeline_evadb.run_pipeline()
    return benchmark_result

def benchmark_movielens_q1_sparkhdoop(num_loop=10, **kwargs):
    movielens_q1_pipeline_sparkhdoop = pipeline.MovielensQ1PipelineSparkHadoop(
        num_loop=num_loop
    )
    benchmark_result = movielens_q1_pipeline_sparkhdoop.run_pipeline()
    return benchmark_result

def benchmark_movielens():
    list_benchmark = []
    # list_benchmark += [benchmark_movielens_q1_dl]
    # list_benchmark += [benchmark_movielens_q1_evadb]
    list_benchmark += [benchmark_movielens_q1_sparkhdoop]


    result_df = None
    num_loop = 1

    
    today = datetime.date.today()
    formatted_date = today.strftime("%m-%d-%Y")

    result_output_file = "result_{}_{}.csv".format('llm', formatted_date)
    
    list_run_config = list_benchmark
    print("[INFO] benchmark config to run: \n \t {}".format(list_run_config))

    for benchmark in tqdm(list_run_config):
        result = benchmark(
            num_loop=num_loop,
        )
        if result_df is None:
            result_df = result
            result.to_csv(result_output_file, sep=",", index=False, mode="a")
        else:
            result.to_csv(result_output_file, sep=",", index=False, mode="a", header=False)


if __name__ == "__main__":
    benchmark_movielens()
