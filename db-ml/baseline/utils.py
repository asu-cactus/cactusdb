import time
import numpy as np
import os


def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def get_postgres_connection_config():
    return "postgresql://postgresdb:postgresdb@localhost:5432/postgresdb"


def convert_df_int64_to_int32(df):
    int64_columns = df.select_dtypes(include=["int64"]).columns
    df[int64_columns] = df[int64_columns].astype(np.int32)
    return df


class Timer(object):
    """A convenient class to measure the running time of a program"""

    def __init__(self):
        self.start = 0
        self.end = 0

    def tic(self):
        """Tic the start time"""
        self.start = time.perf_counter()

    def toc(self):
        """Toc the end time and return the running time

        Returns:
            float: running time (ms)
        """
        self.end = time.perf_counter()
        return (self.end - self.start) * 1000
