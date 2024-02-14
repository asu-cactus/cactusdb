import pipeline
import pandas as pd
import warnings
import load_data_to_db

warnings.filterwarnings("ignore")
from tqdm.auto import tqdm


def benchmark_two_tower_model_pipeline_pytorch(num_sample=500, num_loop=10, **kwargs):
    two_tower_model_pipeline = pipeline.TwoTowerModelPipelineTorch(
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
    )
    benchmark_result = ffnn_pipeline.run_pipeline()
    return benchmark_result


def main():
    # list_benchmark = [benchmark_two_tower_model_pipeline_pytorch, benchmark_two_tower_model_pipeline_tf]
    # list_benchmark = [benchmark_ffnn_pipeline_evadb]
    # list_benchmark += [benchmark_two_tower_model_pipeline_evadb]
    # list_benchmark += [benchmark_two_tower_model_pipeline_evadb]
    list_benchmark = [benchmark_ffnn_pipeline_evadb]
    list_benchmark += [benchmark_ffnn_pipeline_tf, benchmark_ffnn_pipeline_pytorch]
    list_hidden_layer_sizes = [500, 10, 10]
    result_df = None
    num_loop = 10
    num_sample = 500
    num_total_record = 10000
    num_feature = 500
    result_output_file = "result.csv"
    for benchmark in tqdm(list_benchmark):
        if "ffnn" in str(benchmark):
            # needs to generate data
            load_data_to_db.load_ffnn_data_to_postgres(num_total_record, num_feature)
        result = benchmark(
            num_sample=num_sample,
            num_loop=num_loop,
            list_hidden_layer_sizes=list_hidden_layer_sizes,
            num_total_record=num_total_record,
        )
        if result_df is None:
            result_df = result
        else:
            result_df = pd.concat([result_df, result], axis=0)
    result_df.to_csv(result_output_file, sep=",", index=False, mode="a")


if __name__ == "__main__":
    main()
