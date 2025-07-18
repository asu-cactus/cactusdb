import pandas as pd
from pathlib import Path
import pdb

def string_to_int(input_path, output_path):
    # Load the dataset
    df = pd.read_csv(input_path)

    # For any column that is a string, try convert it to an integer
    for col in df.select_dtypes(include=['object']).columns:
        new_col = df[col].transform(lambda x: x.replace("'", ""))
        df[col] = pd.to_numeric(new_col, errors='raise')

       
    # Save the converted DataFrame to a new CSV file
    df.to_csv(output_path, index=False)
    return df

def convert_expedia():
    Path("Expedia/converted").mkdir(parents=True, exist_ok=True)

    input_path = "Expedia/R1_hotels.csv"
    output_path = "Expedia/converted/R1_hotels.csv"
    string_to_int(input_path, output_path)

    input_path = "Expedia/R2_searches.csv"
    output_path = "Expedia/converted/R2_searches.csv"
    string_to_int(input_path, output_path)

    input_path = "Expedia/S_listings_extension.csv"
    output_path = "Expedia/converted/S_listings_extension.csv"
    string_to_int(input_path, output_path)


city_dict = {}
country_dict = {}
dst_dict = {}
active_dict = {}

def process_flights(input_path, output_path):
    # Load the dataset
    df = pd.read_csv(input_path)

    for col in df.select_dtypes(include=['object']).columns:

        if col in ['name2', 'name4', 'codeshare', 'active']:
            # Substitute values "f" -> 0, "t" -> 1
            df[col] = df[col].replace({'f': 0, 't': 1})
        elif col in ['airlineid', 'sairportid', 'dairportid']:
            # Convert to numeric
            df[col] = df[col].transform(lambda x: x.replace("'", ""))
        elif col in ['acountry', 'scountry', 'dcountry']:
            # Build dictionaries for country codes
            global country_dict
            for val in df[col]:
                if val not in country_dict:
                    country_dict[val] = len(country_dict) + 1
            df[col] = df[col].map(country_dict)
        elif col in ['scity', 'dcity']:
            for val in df[col]:
                if val not in city_dict:
                    city_dict[val] = len(city_dict) + 1
            df[col] = df[col].map(city_dict)
        elif col in ['sdst', 'ddst']:
            for val in df[col]:
                if val not in dst_dict:
                    dst_dict[val] = len(dst_dict) + 1
            df[col] = df[col].map(dst_dict)

        # Convert to numeric
        df[col] = pd.to_numeric(df[col], errors='raise')

    # Save the converted DataFrame to a new CSV file
    df.to_csv(output_path, index=False)
    return df

def convert_flights():
    Path("Flights/converted").mkdir(parents=True, exist_ok=True)

    input_path = "Flights/S_routes_first4.csv"
    output_path = "Flights/converted/S_routes_first4.csv"
    process_flights(input_path, output_path)

    input_path = "Flights/R1_airlines.csv"
    output_path = "Flights/converted/R1_airlines.csv"
    process_flights(input_path, output_path)

    input_path = "Flights/R2_sairports.csv"
    output_path = "Flights/converted/R2_sairports.csv"
    process_flights(input_path, output_path)

    input_path = "Flights/R3_dairports.csv"
    output_path = "Flights/converted/R3_dairports.csv"
    process_flights(input_path, output_path)

if __name__ == "__main__":
    # convert_expedia()
    convert_flights()
    print("Conversion completed successfully.")