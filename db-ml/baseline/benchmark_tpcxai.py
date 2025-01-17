import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pipeline
import pandas as pd
import warnings
import load_data_to_db
import itertools
import datetime

warnings.filterwarnings("ignore")
from tqdm.auto import tqdm


def benchmark_tpcxai_usecase03_tf(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase03PipelineTF(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase03_evadb(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase03PipelineEvaDB(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase03_sparkhadoop(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase3PipelineSparkHadoop(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase03_madlib(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase03PipelineMadlib(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase08_tf(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase08PipelineTF(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase08_evadb(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase08PipelineEvaDB(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase08_sparkhadoop(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase8PipelineSparkHadoop(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase08_madlib(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase08PipelineMadlib(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase10_tf(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase10PipelineTF(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase10_ml_tf(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase10MLPipelineTF(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase10_evadb(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase10PipelineEvaDB(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase10_ml_evadb(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase10MLPipelineEvaDB(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase10_sparkhadoop(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase10PipelineSparkHadoop(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase10_ml_sparkhadoop(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase10PipelineSparkMLHadoop(
        num_loop=num_loop
    )
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase10_madlib(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase10PipelineMadlib(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai_usecase10_ml_madlib(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.TPCxAIUsecase10MLPipelineMadlib(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_tpcxai():
    list_benchmark = []
    # list_benchmark += [benchmark_tpcxai_usecase03_tf]
    # list_benchmark += [benchmark_tpcxai_usecase03_evadb]
    # list_benchmark += [benchmark_tpcxai_usecase03_sparkhadoop]
    # list_benchmark += [benchmark_tpcxai_usecase03_madlib]
    # list_benchmark += [benchmark_tpcxai_usecase08_tf]
    # list_benchmark += [benchmark_tpcxai_usecase08_evadb]
    # list_benchmark += [benchmark_tpcxai_usecase08_sparkhadoop]
    # list_benchmark += [benchmark_tpcxai_usecase08_madlib]
    # list_benchmark += [benchmark_tpcxai_usecase10_tf]
    # list_benchmark += [benchmark_tpcxai_usecase10_evadb]
    # list_benchmark += [benchmark_tpcxai_usecase10_sparkhadoop]
    # list_benchmark += [benchmark_tpcxai_usecase10_madlib]
    list_benchmark += [benchmark_tpcxai_usecase10_ml_tf]
    list_benchmark += [benchmark_tpcxai_usecase10_ml_evadb]
    list_benchmark += [benchmark_tpcxai_usecase10_ml_sparkhadoop]
    list_benchmark += [benchmark_tpcxai_usecase10_ml_madlib]
    list_num_user = [1]
    list_num_movie = [1]

    result_df = None
    num_loop = 4

    today = datetime.date.today()
    formatted_date = today.strftime("%m-%d-%Y")

    result_output_file = "result_{}_{}.csv".format("tpcxai", formatted_date)

    list_run_config = list(
        itertools.product(list_num_user, list_num_movie, list_benchmark)
    )
    print("[INFO] benchmark config to run: \n \t {}".format(list_run_config))

    for num_user, num_movie, benchmark in tqdm(list_run_config):
        result = benchmark(
            num_user=num_user,
            num_movie=num_movie,
            num_loop=num_loop,
        )
        result["num_user"] = str(num_user)
        result["num_movie"] = str(num_movie)
        if result_df is None:
            result_df = result
            result.to_csv(result_output_file, sep=",", index=False, mode="a")
        else:
            result.to_csv(
                result_output_file, sep=",", index=False, mode="a", header=False
            )


if __name__ == "__main__":
    benchmark_tpcxai()
