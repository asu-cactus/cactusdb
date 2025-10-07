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


def benchmark_llm_recommendation_pipeline_python(
    num_user, num_movie, num_loop=10, **kwargs
):
    llm_recommendation_pipeline = pipeline.LLMRecommendationPipelinePython(
        num_user=num_user, num_movie=num_movie, num_loop=num_loop
    )
    benchmark_result = llm_recommendation_pipeline.run_pipeline()
    return benchmark_result


def benchmark_llm_recommendation_pipeline2_python(
    num_user, num_movie, num_loop=10, **kwargs
):
    llm_recommendation_pipeline = pipeline.LLMRecommendationPipeline2Python(
        num_user=num_user, num_movie=num_movie, num_loop=num_loop
    )
    benchmark_result = llm_recommendation_pipeline.run_pipeline()
    return benchmark_result


def benchmark_llm_movie_info_retrieval_pipeline_python(
    num_user, num_movie, num_loop=10, **kwargs
):
    llm_recommendation_pipeline = pipeline.LLMMovieInfoRetrievalPipelinePython(
        num_user=num_user, num_movie=num_movie, num_loop=num_loop
    )
    benchmark_result = llm_recommendation_pipeline.run_pipeline()
    return benchmark_result


def benchmark_llm_movie_info_retrieval_pipeline_blendsql(
    num_user, num_movie, num_loop=10, **kwargs
):
    llm_recommendation_pipeline = pipeline.LLMMovieInfoRetrievalPipelineBlendSQL(
        num_user=num_user, num_movie=num_movie, num_loop=num_loop
    )
    benchmark_result = llm_recommendation_pipeline.run_pipeline()
    return benchmark_result


def benchmark_llm():
    list_benchmark = []
    # list_benchmark += [benchmark_llm_recommendation_pipeline_python]
    # list_benchmark += [benchmark_llm_recommendation_pipeline2_python]
    list_benchmark += [benchmark_llm_movie_info_retrieval_pipeline_python]
    list_benchmark += [benchmark_llm_movie_info_retrieval_pipeline_blendsql]
    list_num_user = [50]
    list_num_movie = [200]

    result_df = None
    num_loop = 1

    today = datetime.date.today()
    formatted_date = today.strftime("%m-%d-%Y")

    result_output_file = "result_{}_{}.csv".format("llm", formatted_date)

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
    benchmark_llm()
