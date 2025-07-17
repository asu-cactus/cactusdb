import utils
import psycopg2

# Function to execute query
def execute_query(query, params=None, fetch=False):
    """Execute a query in PostgreSQL and return results if needed."""
    conn = None
    try:
        # Connect to the database
        conn = utils.get_psycopg2_connection()
        cur = conn.cursor()
        cur.execute(query, params)
        
        # Fetch results if needed
        result = cur.fetchmany(10) if fetch else None
        
        # Commit and close
        conn.commit()
        cur.close()
        return result
    except psycopg2.Error as e:
        print(f"Error executing query: {e}")
    finally:
        if conn:
            conn.close()

def load_uc03_training_data():
    query_to_fetch_training_data = """
    DROP VIEW IF EXISTS imbridge3_training_data;
    CREATE OR REPLACE VIEW imbridge3_training_data as (
        SELECT Class::INT, V1::real, V2::real, V3::real, V4::real, V5::real, V6::real, V7::real, V8::real, V9::real, V10::real, V11::real, V12::real, V13::real, V14::real, V15::real, V16::real, V17::real, V18::real, V19::real, V20::real, V21::real, V22::real, V23::real, V24::real, V25::real, V26::real, V27::real, V28::real, Amount::real
        FROM Credit_Card_extension 
        );
    """

    # Prepare training data
    utils.execute_sql_query_via_psycopg2(query_to_fetch_training_data)


def load_uc03_serving_data():
    query_to_fetch_serving_data = """
    DROP VIEW IF EXISTS imbridge3_serving_data;
    CREATE OR REPLACE VIEW imbridge3_serving_data as (
        SELECT ARRAY [ V1::real, V2::real, V3::real, V4::real, V5::real, V6::real, V7::real, V8::real, V9::real, V10::real, V11::real, V12::real, V13::real, V14::real, V15::real, V16::real, V17::real, V18::real, V19::real, V20::real, V21::real, V22::real, V23::real, V24::real, V25::real, V26::real, V27::real, V28::real, Amount::real] AS features
        FROM Credit_Card_extension
        );
    """
    # Prepare serving data
    utils.execute_sql_query_via_psycopg2(query_to_fetch_serving_data)


def train_uc03_model():
    execute_query("""
CREATE EXTENSION IF NOT EXISTS pgml;

SELECT pgml.train(
    project_name => 'imbridge3_model_xgboost',
    task => 'classification',
    relation_name => 'imbridge3_training_data',
    y_column_name => 'class',
    algorithm => 'xgboost',
    hyperparams => '{
       "n_estimators": 100, "max_depth": 9
   }'::JSONB
);
                                                     
    """)

    results = utils.fetch_data_from_postgres_via_psycopg2("""
            SELECT * FROM pgml.deployed_models;
        """)
    if results is not None:
        print(results)
    else:
        print("No models.")


def test_uc03_model():
    result_serving = utils.fetch_data_from_postgres_via_psycopg2("""
SET max_parallel_workers_per_gather = 4;
SELECT pgml.predict_batch('imbridge3_model_xgboost', array_agg(features)) as prediction FROM imbridge3_serving_data;
    """)

    if result_serving is not None:
        print("Model tested successfully.")
        print(f"Number of predictions: {len(result_serving)}")
        print(result_serving.head())
    else:
        print("Model testing failed.")

                                                              
if __name__ == "__main__":
    load_uc03_training_data()
    train_uc03_model()
    load_uc03_serving_data()
    test_uc03_model()
  