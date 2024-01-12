import pipeline
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from tqdm.auto import tqdm


def benchmark_two_tower_model_pipeline_pytorch(num_sample=500, num_loop=10):
    two_tower_model_pipeline = pipeline.TwoTowerModelPipelineTorch(num_sample=num_sample, num_loop=num_loop)
    benchmark_result = two_tower_model_pipeline.run_pipeline()
    return benchmark_result

def benchmark_two_tower_model_pipeline_tf(num_sample=500, num_loop=10):
    two_tower_model_pipeline = pipeline.TwoTowerModelPipelineTF(num_sample=num_sample, num_loop=num_loop)
    benchmark_result = two_tower_model_pipeline.run_pipeline()
    return benchmark_result

def benchmark_two_tower_model_pipeline_evadb(num_sample=500, num_loop=10):
    two_tower_model_pipeline = pipeline.TwoTowerModelPipelineEvaDB(num_sample=num_sample, num_loop=num_loop)
    benchmark_result = two_tower_model_pipeline.run_pipeline()
    return benchmark_result

def benchmark_ffnn_pipeline_evadb(num_sample=500, num_loop=10):
    ffnn_pipeline = pipeline.FFNNEvaDB(num_sample=num_sample, num_loop=num_loop)
    benchmark_result = ffnn_pipeline.run_pipeline()
    return benchmark_result

def main():
    list_benchmark = [benchmark_two_tower_model_pipeline_pytorch, benchmark_two_tower_model_pipeline_tf]
    # list_benchmark = [benchmark_ffnn_pipeline_evadb]
    list_benchmark += [benchmark_two_tower_model_pipeline_evadb]
    result_df = None
    num_loop = 10
    num_sample = 50000
    result_output_file = 'result.csv'
    for benchmark in tqdm(list_benchmark):
        result = benchmark(num_sample=num_sample, num_loop=num_loop)
        if result_df is None:
            result_df = result 
        else:
            result_df = pd.concat([result_df, result], axis=0)
    result_df.to_csv(result_output_file, sep=',', index=False, mode='a')

if __name__ == "__main__":
    main()