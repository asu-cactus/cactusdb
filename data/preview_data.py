import pyarrow.parquet as pq 

print(pq.ParquetFile('/root/velox_latest/data/query_data.parquet').metadata)