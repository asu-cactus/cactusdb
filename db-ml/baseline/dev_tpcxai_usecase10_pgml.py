import psycopg2
import utils

# Function to create PGML Extension
def create_pgml_extension():
    """Creates the pgml extension in PostgreSQL"""
    conn = None
    try:
        conn = utils.get_psycopg2_connection()
        cur = conn.cursor()
        
        # Create the extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgml;")
        
        conn.commit()
        print("PGML extension created successfully!")
        cur.close()
    except psycopg2.Error as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

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

# Create PGML Extension if it does not exists
create_pgml_extension()

# prepare training data
query_to_fetch_training_data = """
create view uc10_training_data as select EXTRACT(HOUR FROM time) / 23 as business_hour_norm, amount / transaction_limit as amount_norm, is_fraud
from tpcxai_financial_account_training join tpcxai_financial_transactions_training on fa_customer_sk=sender_id
"""

execute_query(query_to_fetch_training_data)


# Train Model
execute_query("""
    SELECT pgml.train(
        project_name => 'uc10_logistic_model',
        task => 'classification',
        relation_name => 'uc10_training_data',
        y_column_name => 'is_fraud',
        algorithm => 'linear'
    );
""")
print("Model Trained successfully")


# Prepare testing data
query_to_fetch_serving_data = """
create or replace view uc10_serving_data as select transaction_id, ARRAY [(EXTRACT(HOUR FROM time) / 23)::real, (amount / transaction_limit)::real] AS features from tpcxai_financial_account_serving join tpcxai_financial_transactions_serving on fa_customer_sk=sender_id
"""
execute_query("DROP VIEW IF EXISTS uc10_serving_data;")
execute_query(query_to_fetch_serving_data)

# Predict on test data
prediction = execute_query("""
    SELECT transaction_id, pgml.predict('uc10_logistic_model', features) as prediction from uc10_serving_data;
""", fetch=True)

print("Prediction result:", prediction)




