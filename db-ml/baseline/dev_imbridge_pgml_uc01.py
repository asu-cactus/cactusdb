import utils
from dev_imbridge_pgml_uc03 import execute_query

def load_uc01_training_data():
    # utils.execute_sql_query_via_psycopg2("""
    #     DROP TABLE IF EXISTS Expedia_S_listings_extension CASCADE;
    #     CREATE TABLE Expedia_S_listings_extension (
    #         srch_id VARCHAR(10),
    #         prop_id VARCHAR(10),
    #         position VARCHAR(3),
    #         prop_location_score1 FLOAT,
    #         prop_location_score2 FLOAT,
    #         prop_log_historical_price FLOAT,
    #         price_usd FLOAT,
    #         promotion_flag INT,
    #         orig_destination_distance FLOAT);
    #     COPY Expedia_S_listings_extension FROM '/home/imbridge_data/IMBridge_exp/public_datasets/Expedia/S_listings_extension.csv' DELIMITER ',' CSV HEADER;


    #     DROP TABLE IF EXISTS Expedia_R1_hotels CASCADE;
    #     CREATE TABLE Expedia_R1_hotels (
    #         prop_id VARCHAR(10),
    #         prop_country_id VARCHAR(10),
    #         prop_starrating INT,
    #         prop_review_score FLOAT,
    #         prop_brand_bool INT,
    #         count_clicks INT,
    #         avg_bookings_usd FLOAT,
    #         stdev_bookings_usd FLOAT,
    #         count_bookings INT);
    #     COPY Expedia_R1_hotels FROM '/home/imbridge_data/IMBridge_exp/public_datasets/Expedia/R1_hotels.csv' DELIMITER ',' CSV HEADER;

    
    #     DROP TABLE IF EXISTS Expedia_R2_searches CASCADE;
    #     CREATE TABLE Expedia_R2_searches (
    #         srch_id VARCHAR(10),
    #         year VARCHAR(6),
    #         month VARCHAR(4),
    #         weekofyear VARCHAR(4),
    #         time VARCHAR(10),
    #         site_id VARCHAR(6),
    #         visitor_location_country_id VARCHAR(6),
    #         srch_destination_id VARCHAR(8),
    #         srch_length_of_stay INT,
    #         srch_booking_window INT,
    #         srch_adults_count INT,
    #         srch_children_count INT,
    #         srch_room_count INT,
    #         srch_saturday_night_bool INT,
    #         random_bool INT
    #     );
    #     COPY Expedia_R2_searches FROM '/home/imbridge_data/IMBridge_exp/public_datasets/Expedia/R2_searches.csv' DELIMITER ',' CSV HEADER;
    # """)
        
    query_to_fetch_training_data = """
    DROP VIEW IF EXISTS imbridge1_training_data;
    CREATE OR REPLACE VIEW imbridge1_training_data as (
        SELECT promotion_flag::int, prop_location_score1::real, prop_location_score2::real, prop_log_historical_price::real, price_usd::real,
                orig_destination_distance::real, prop_review_score::real, avg_bookings_usd::real, stdev_bookings_usd::real,
                position::int, prop_country_id::int, prop_starrating::int, prop_brand_bool::int, count_clicks::int, count_bookings::int,
                year::int, month::int, weekofyear::int, time::int, site_id::int, visitor_location_country_id::int, srch_destination_id::int,
                srch_length_of_stay::int, srch_booking_window::int, srch_adults_count::int, srch_children_count::int,
                srch_room_count::int, srch_saturday_night_bool::int, random_bool::int
        FROM Expedia_S_listings_extension JOIN Expedia_R1_hotels ON Expedia_S_listings_extension.prop_id = Expedia_R1_hotels.prop_id
        JOIN Expedia_R2_searches ON Expedia_S_listings_extension.srch_id = Expedia_R2_searches.srch_id
        WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1
        and prop_log_historical_price > 4 and count_bookings > 5
        and srch_booking_window > 10 and srch_length_of_stay > 1
        );
    """

    # Prepare training data
    utils.execute_sql_query_via_psycopg2(query_to_fetch_training_data)

def load_uc01_serving_data():
    query_to_fetch_serving_data = """
    DROP VIEW IF EXISTS imbridge1_serving_data;
    CREATE OR REPLACE VIEW imbridge1_serving_data as (
        SELECT ARRAY [ prop_location_score1::real, prop_location_score2::real, prop_log_historical_price::real, price_usd::real,
                orig_destination_distance::real, prop_review_score::real, avg_bookings_usd::real, stdev_bookings_usd::real,
                position::int, prop_country_id::int, prop_starrating::int, prop_brand_bool::int, count_clicks::int, count_bookings::int,
                year::int, month::int, weekofyear::int, time::int, site_id::int, visitor_location_country_id::int, srch_destination_id::int,
                srch_length_of_stay::int, srch_booking_window::int, srch_adults_count::int, srch_children_count::int,
                srch_room_count::int, srch_saturday_night_bool::int, random_bool::int ] AS features
        FROM Expedia_S_listings_extension JOIN Expedia_R1_hotels ON Expedia_S_listings_extension.prop_id = Expedia_R1_hotels.prop_id
        JOIN Expedia_R2_searches ON Expedia_S_listings_extension.srch_id = Expedia_R2_searches.srch_id
        WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1
        and prop_log_historical_price > 4 and count_bookings > 5
        and srch_booking_window > 10 and srch_length_of_stay > 1
        );
    """
    # Prepare serving data
    utils.execute_sql_query_via_psycopg2(query_to_fetch_serving_data)

def train_uc01_model():
    result_train = execute_query("""
CREATE EXTENSION IF NOT EXISTS pgml;
SELECT pgml.train(
    project_name => 'imbridge1_model_classification',
    task => 'classification',
    relation_name => 'imbridge1_training_data',
    y_column_name => 'promotion_flag',
    algorithm => 'random_forest',
    hyperparams => '{
       "n_estimators": 1, "max_depth": 9
   }'::JSONB
);
                                                     
    """)
    if result_train is not None:
        print("Model trained successfully.")
        print(result_train)
        
#         # Deploy the model
#         result_deploy = utils.fetch_data_from_postgres_via_psycopg2("""
# SELECT * FROM pgml.deploy(
#     project_name => 'imbridge1_model',
#     strategy => 'most_recent'
# );
#         """)
        
#         if result_deploy is not None:
#             print("Model deployed successfully.")
#             print(result_deploy)
#         else:
#             print("Model deployment failed.")
    else:
        print("Model training failed.")

def test_uc01_model():
    result_serving = utils.fetch_data_from_postgres_via_psycopg2("""
SET max_parallel_workers_per_gather = 0;
SELECT pgml.predict_batch('imbridge1_model_classification', array_agg(features)) as prediction FROM imbridge1_serving_data;
    """)

    if result_serving is not None:
        print("Model tested successfully.")
        print(f"Number of predictions: {len(result_serving)}")
        print(result_serving.head())
    else:
        print("Model testing failed.")

                                                         
if __name__ == "__main__":
    # load_uc01_training_data()
    # train_uc01_model()
    load_uc01_serving_data()
    test_uc01_model()
  