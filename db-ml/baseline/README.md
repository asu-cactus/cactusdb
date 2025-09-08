
## Data Loading

Before running the baseline benchmarks for different workloads, you need to load the data into the datastore. Use the following command:

```bash
cd db-ml/baseline
# Use the --dataset flag to specify the dataset you want to load,
# or pass 'all' to load every dataset.
python load_data_to_db.py --dataset=XX  # Options: tpcxai, movielens_recommendation, etc.
```


## Running Benchmarks

- MovieLens workloads: Run the following command with the desired workload queries listed in list_benchmark: `python benchmark_movielens.py`

- TPCx-AI workloads: Run the following command with the desired workload queries listed in list_benchmark: `python benchmark_tpcxai.py`


## Troubleshooting

If GPU support is enabled, install TensorFlow with CUDA support:

```bash
pip install "tensorflow[and-cuda]==2.15" --extra-index-url https://pypi.nvidia.com
```