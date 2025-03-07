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

query_create_func_dept_enc = """
CREATE OR REPLACE FUNCTION uc8_department_encoder(category TEXT)
          RETURNS INT AS $$
          DECLARE
              category_list TEXT[] := ARRAY[
                  'AUTOMOTIVE', 'BATH AND SHOWER', 'BEAUTY', 'BEDDING', 'BOYS WEAR',
                  'CANDY, TOBACCO, COOKIES', 'CELEBRATION', 'COMM BREAD',
                  'COOK AND DINE', 'DAIRY', 'DSD GROCERY', 'ELECTRONICS',
                  'FABRICS AND CRAFTS', 'FINANCIAL SERVICES', 'FROZEN FOODS',
                  'GIRLS WEAR, 4-6X  AND 7-14', 'GROCERY DRY GOODS', 'HARDWARE',
                  'HOME DECOR', 'HOME MANAGEMENT', 'HORTICULTURE AND ACCESS',
                  'HOUSEHOLD CHEMICALS/SUPP', 'HOUSEHOLD PAPER GOODS',
                  'IMPULSE MERCHANDISE', 'INFANT APPAREL',
                  'INFANT CONSUMABLE HARDLINES', 'JEWELRY AND SUNGLASSES',
                  'LADIESWEAR', 'LAWN AND GARDEN', 'LIQUOR,WINE,BEER',
                  'MEAT - FRESH & FROZEN', 'MEDIA AND GAMING', 'MENS WEAR',
                  'OFFICE SUPPLIES', 'PAINT AND ACCESSORIES', 'PERSONAL CARE',
                  'PETS AND SUPPLIES', 'PHARMACY OTC', 'PHARMACY RX',
                  'PLAYERS AND ELECTRONICS', 'PRODUCE', 'SERVICE DELI', 'SHOES',
                  'SPORTING GOODS', 'TOYS', 'WIRELESS'
              ];
              index INT;
          BEGIN
              -- Find the index of the category in the list
              index := array_position(category_list, category);
              
              -- If not found, return -1
              IF index IS NULL THEN
                  RETURN -1;
              ELSE
                  RETURN index - 1; -- Convert 1-based index to 0-based index
              END IF;
          END;
          $$ LANGUAGE plpgsql;
"""

query_create_func_trip_type_enc = """
CREATE OR REPLACE FUNCTION uc8_trip_type_encoder(value NUMERIC)
RETURNS INTEGER AS $$
DECLARE
    predefined_values NUMERIC[] := ARRAY[3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 18, 19, 20,
                                         21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                         31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                         41, 42, 43, 44, 999];
    index INTEGER;
BEGIN
    -- Find the index of the value in the predefined list
    index := array_position(predefined_values, value);
    
    -- If the value doesn't exist, return -1 or NULL
    IF index IS NULL THEN
        RETURN -1;  -- Change to `NULL` if you want to return NULL instead of -1
    END IF;

    RETURN index - 1;
END;
$$ LANGUAGE plpgsql;
"""

# Create department and trip type encoding functions
execute_query(query_create_func_dept_enc)
execute_query(query_create_func_trip_type_enc)

# prepare training data
query_to_fetch_training_data = """
CREATE OR REPLACE VIEW uc8_training_data as (
  SELECT department, quantity, scan_count, weekday, uc8_trip_type_encoder(trip_type) AS trip_type
  FROM (
    SELECT
        o_order_id,
        uc8_department_encoder(department) AS department,
        quantity,
        SUM(quantity) AS scan_count,
        MIN(EXTRACT(DOW FROM date)) AS weekday,
        MIN(trip_type) AS trip_type
    FROM tpcxai_order_training
    JOIN tpcxai_lineitem_training ON o_order_id = li_order_id
    JOIN tpcxai_product_training ON li_product_id = p_product_id
    GROUP BY o_order_id, date, department, quantity
  ) as t
ORDER BY random()
LIMIT 100000
);
"""
execute_query(query_to_fetch_training_data)


# Train Model
execute_query("""
    SELECT pgml.train(
        project_name => 'uc8_xgboost_model',
        task => 'classification',
        relation_name => 'uc8_training_data',
        y_column_name => 'trip_type',
        algorithm => 'xgboost',
        test_size => 0.5,
        test_sampling => 'last',
        hyperparams => '{
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 50
        }'
    );
""")
print("Model Trained successfully")


# Prepare testing data
query_to_fetch_serving_data = """
CREATE OR REPLACE VIEW uc8_serving_data as (
              SELECT o_order_id, ARRAY [(department)::real, (quantity)::real, (quantity)::real, (weekday)::real] AS features
              FROM
                  (
                    SELECT
                    o_order_id,
                    uc8_department_encoder(department) as department,
                    quantity,
                    SUM(quantity) AS scan_count,
                    MIN(EXTRACT(DOW FROM date)) AS weekday
                  FROM tpcxai_order_serving
                  JOIN tpcxai_lineitem_serving ON o_order_id = li_order_id
                  JOIN tpcxai_product_serving ON li_product_id = p_product_id
                  GROUP BY o_order_id, date, department, quantity
                  ) as t
          );
"""
execute_query(query_to_fetch_serving_data)

# Predict on test data
prediction = execute_query("""
    SELECT o_order_id, pgml.predict('uc8_xgboost_model', features) as prediction from uc8_serving_data;
""", fetch=True)

print("Prediction result:", prediction)
