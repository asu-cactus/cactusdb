import utils
from dev_imbridge_pgml_uc03 import execute_query

def load_uc02_training_data():
    query_to_fetch_training_data = """
    DROP VIEW IF EXISTS imbridge2_training_data;
    CREATE OR REPLACE VIEW imbridge2_training_data as (
        SELECT codeshare::int, slatitude::real, slongitude::real, dlatitude::real, dlongitude::real, name1::int, name2::int, name4::int, acountry::int, active::int,
                scity::int, scountry::int, stimezone::int, sdst::int, dcity::int, dcountry::int, dtimezone::int, ddst::int
        FROM flights_s_routes_extension JOIN flights_r1_airlines ON flights_s_routes_extension.airlineid = flights_r1_airlines.airlineid
        JOIN flights_r2_sairports ON flights_s_routes_extension.sairportid = flights_r2_sairports.sairportid JOIN flights_r3_dairports
        ON flights_s_routes_extension.dairportid = flights_r3_dairports.dairportid
        WHERE name2 = 1 and name4 = 1 and name1 > 2.8
    );
    """

    # Prepare training data
    utils.execute_sql_query_via_psycopg2(query_to_fetch_training_data)


def load_uc02_serving_data():
    query_to_fetch_serving_data = """
    DROP VIEW IF EXISTS imbridge2_serving_data;
    CREATE OR REPLACE VIEW imbridge2_serving_data as (
        SELECT ARRAY [ slatitude::real, slongitude::real, dlatitude::real, dlongitude::real, name1::int, name2::int, name4::int, acountry::int, active::int,
                    scity::int, scountry::int, stimezone::int, sdst::int, dcity::int, dcountry::int, dtimezone::int, ddst::int ] AS features
            FROM flights_s_routes_extension JOIN flights_r1_airlines ON flights_s_routes_extension.airlineid = flights_r1_airlines.airlineid
            JOIN flights_r2_sairports ON flights_s_routes_extension.sairportid = flights_r2_sairports.sairportid JOIN flights_r3_dairports
            ON flights_s_routes_extension.dairportid = flights_r3_dairports.dairportid
            WHERE name2 = 1 and name4 = 1 and name1 > 2.8
    );
    """
    # Prepare serving data
    utils.execute_sql_query_via_psycopg2(query_to_fetch_serving_data)


def train_uc02_model():
    execute_query("""
CREATE EXTENSION IF NOT EXISTS pgml;

SELECT pgml.train(
    project_name => 'imbridge2_model_rf',
    task => 'classification',
    relation_name => 'imbridge2_training_data',
    y_column_name => 'codeshare',
    algorithm => 'random_forest',
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


def test_uc02_model():
    result_serving = utils.fetch_data_from_postgres_via_psycopg2("""
SET max_parallel_workers_per_gather = 4;
SELECT pgml.predict_batch('imbridge2_model_rf', array_agg(features)) as prediction FROM imbridge2_serving_data;
    """)

    if result_serving is not None:
        print("Model tested successfully.")
        print(f"Number of predictions: {len(result_serving)}")
        print(result_serving.head())
    else:
        print("Model testing failed.")

                                                              
if __name__ == "__main__":
    # load_uc02_training_data()
    # train_uc02_model()
    load_uc02_serving_data()
    test_uc02_model()
