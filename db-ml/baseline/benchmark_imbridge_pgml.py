import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pipeline
# import pandas as pd
import warnings
# import load_data_to_db
# import itertools
import datetime

warnings.filterwarnings("ignore")
from tqdm.auto import tqdm


def benchmark_imbridge_usecase01_pgml(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.ImbridgeUsecase01PipelinePGML(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result

def benchmark_imbridge_usecase02_pgml(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.ImbridgeUsecase02PipelinePGML(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result

def benchmark_imbridge_usecase03_pgml(num_loop=10, **kwargs):
    benchmark_pipeline = pipeline.ImbridgeUsecase03PipelinePGML(num_loop=num_loop)
    benchmark_result = benchmark_pipeline.run_pipeline()
    return benchmark_result


def benchmark_imbridge():
    list_benchmark = [benchmark_imbridge_usecase01_pgml, benchmark_imbridge_usecase02_pgml, benchmark_imbridge_usecase03_pgml]
    # list_benchmark = [benchmark_imbridge_usecase02_pgml]
    result_df = None
    num_loop = 10

    today = datetime.date.today()
    formatted_date = today.strftime("%m-%d-%Y")

    result_output_file = f"result_imbridge_{formatted_date}.csv"

    print("[INFO] benchmark config to run: \n \t {}".format(list_benchmark))

    for  benchmark in tqdm(list_benchmark):
        result = benchmark(
            num_loop=num_loop
        )

        if result_df is None:
            result_df = result
            result.to_csv(result_output_file, sep=",", index=False, mode="a")
        else:
            result.to_csv(
                result_output_file, sep=",", index=False, mode="a", header=False
            )


if __name__ == "__main__":
    benchmark_imbridge()
