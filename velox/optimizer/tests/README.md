## Model2Vec and Query2Vec

The implementations of Model2Vec and Query2Vec are located in the [python](../python/) folder. Benchmarking with generated models and queries is required to prepare training data for both components.

1. Run the following scripts to collect training data:
  - `benchmark_with_model_template.py` → for Model2Vec
  - `benchmark_with_query_template.py` → for Query2Vec
  - Each run will generate a timestamped log file in the generatedQueryPlan folder.

Run the `benchmark_with_model_template.py` and `benchmark_with_query_template.py` to collect training data for Model2Vec and Query2Vec. Each benchmarking will output a log file with timestamp generated to `generatedQueryPlan` folder.

2. Train the embedding models using the provided notebooks:
  - [Model2Vec Training](../python/Model2VecTraining.ipynb) 
  - [Query2Vec Training](../python/Query2VecTraining.ipynb)
  - Update the input data path by setting the correct timestamp of your generated logs.

## Benchmark with Reusable MCTS

After training Model2Vec and Query2Vec, you can benchmark the query optimizer with Reusable MCTS.

- Run the Python entry point: `python ReusableMCTS.py`
- Run the C++ test driver: `./ReusableMCTSTest --mcts=<heuristic|arbitrary|vanilla|reusable> --query_template=<ml|tpcxai>`

Here: 
optimizer specifies the MCTS variant to use (heuristic, arbitrary, vanilla, or reusable).

query_template specifies the workload (ml = MovieLens, tpcxai = TPCx-AI).

