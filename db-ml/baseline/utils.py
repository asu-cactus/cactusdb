import time
import numpy as np
import psycopg2
import pandas as pd
import os
import evadb
import multiprocessing
import subprocess
import connectorx as cx


def get_sys_num_threads():
    return multiprocessing.cpu_count()


def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def get_postgres_connection_config():
    return "postgresql://postgresdb:postgresdb@localhost:5432/postgresdb"


def get_jdbc_postgres_connection_config():
    return "jdbc:postgresql://localhost:5432/postgresdb"


def get_sparksql_postgres_connection_properties():
    return {
        "user": "postgresdb",
        "password": "postgresdb",
        "driver": "org.postgresql.Driver",
    }

def get_connectorx_configuration():
    db_params = {
        "dbname": "postgresdb",
        "user": "postgresdb",
        "password": "postgresdb",
        "host": "localhost",
        "port": "5432",
    }
    
    return db_params

def get_psycopg2_connection():
    db_params = {
        "dbname": "postgresdb",
        "user": "postgresdb",
        "password": "postgresdb",
        "host": "localhost",
        "port": "5432",
    }
    try:
        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(**db_params)
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    return connection


def fetch_data_from_postgres_via_connectorx(sql):
    # Database connection parameters
    db_conn = get_postgres_connection_config()

    try:
        df = cx.read_sql(db_conn, sql)         

    except Exception as e: 
        print(f"Error: {e}")
    return df


# TODO: in the future, fetching results from DB should utilize connectorx
# for better performance, while connectorx can only execute one query at once
def fetch_data_from_postgres_via_psycopg2(command):
    # Database connection parameters
    db_params = {
        "dbname": "postgresdb",
        "user": "postgresdb",
        "password": "postgresdb",
        "host": "localhost",
        "port": "5432",
    }

    try:
        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(**db_params)

        # Create a cursor object
        cursor = connection.cursor()

        # Execute the drop view command
        cursor.execute(command)

        result_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(result_data, columns=column_names)
        # Commit the transaction

        # Close the cursor and connection
        cursor.close()
        connection.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    return df


def convert_df_int64_to_int32(df):
    int64_columns = df.select_dtypes(include=["int64"]).columns
    df[int64_columns] = df[int64_columns].astype(np.int32)
    return df

def check_hdfs_dir_exist(directory_path):
  """
  This function checks if a directory exists in HDFS.

  Args:
      directory_path (str): The path to the directory in HDFS.

  Returns:
      bool: True if the directory exists, False otherwise.
  """
  command = ["hdfs", "dfs", "-test", "-d", directory_path]
  result = subprocess.run(command, capture_output=True)
  print(result)
  return result.returncode == 0

def create_hdfs_dir(directory_path):
  """
  """
  command = ["hdfs", "dfs", "-mkdir", "-p", directory_path]
  result = subprocess.run(command, capture_output=True)
  print(result)

def load_csv_to_hdfs(src_path, tar_path):
  """
  """
  command = ["hdfs", "dfs", "-put", src_path, tar_path]
  result = subprocess.run(command, capture_output=True)
  print(result)

def ls_hdfs_dir(directory_path):
  """
  """
  command = ["hdfs", "dfs", "-ls", "-C", directory_path]
  result = subprocess.run(command, capture_output=True)
  return result

def rm_hdfs_file(path):
  """
  """
  command = ["hdfs", "dfs", "-rm", "-r", path]
  result = subprocess.run(command, capture_output=True)
#   print(result)

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


def setup_postgres_for_evadb():
    params = {
        "user": "postgresdb",
        "password": "postgresdb",
        "host": "localhost",
        "port": "5432",
        "database": "postgresdb",
    }
    query = f"CREATE DATABASE IF NOT EXISTS postgres_data WITH ENGINE = 'postgres', PARAMETERS = {params};"
    cursor = evadb.connect().cursor()
    cursor.query(query).df()

def check_table_exist(cursor, table_name):
    cursor.execute("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)", (table_name,))
    exists = cursor.fetchone()[0]
    return exists