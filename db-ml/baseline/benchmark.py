import pipeline
import pandas as pd
from tqdm.auto import tqdm


def benchmark_two_tower_model_pipeline(num_loop=10):
    two_tower_model_pipeline = pipeline.TwoTowerModelPipeline(num_loop)
    benchmark_result = two_tower_model_pipeline.run_pipeline()
    return benchmark_result

def main():
    list_benchmark = [benchmark_two_tower_model_pipeline]
    result_df = None
    num_loop=10
    result_output_file = 'result.csv'
    for benchmark in tqdm(list_benchmark):
        print("debug", benchmark)
        result = benchmark(num_loop)
        if result_df is None:
            result_df = result 
        else:
            result_df = pd.concat([result_df, result], axis=0)
    result_df.to_csv(result_output_file, sep=',', index=False, mode='a')

if __name__ == "__main__":
    main()