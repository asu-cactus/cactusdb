import numpy as np
import pandas as pd
import random
import os
import json


def get_join_statistics(join_id, left_table_name, right_table_name, left_table_actual, left_key, right_key, projection, left_rows, left_cols, right_rows, right_cols):
    join_info = {}
    join_info["ID"] = join_id
    join_info["Left"] = left_table_name
    join_info["Right"] = right_table_name
    join_info["Pred"] = f"{left_table_actual}.{left_key} = {right_table_name}.{right_key}"
    join_info["ProbeKeys"] = left_key
    join_info["BuildKeys"] = right_key
    join_info["Projection"] = projection
    join_info["NumTuplesLeft"] = left_rows
    join_info["NumDimLeft"] = left_cols
    join_info["NumTuplesRight"] = right_rows
    join_info["NumDimRight"] = right_cols
    join_info["NumTuplesOutput"] = right_rows
    join_info["NumDimOutput"] = left_cols + right_cols
    return join_info


# Function to generate a random table with size controls
def generate_table_optimized(table_idx, join_key, num_rows, num_columns, ratio=None, join_key_values=None):
    if join_key_values is None:
        join_key_values = np.random.randint(1, num_rows + 1, size=num_rows)
    else:
        #ratio = 5  # Limit the duplication factor to control row growth
        #num_rows = min(num_rows, len(join_key_values) * ratio)
        #join_key_values = np.repeat(join_key_values, np.random.randint(1, ratio + 1))[:num_rows]
        if ratio < 1:
            join_key_values = np.random.choice(join_key_values, size=num_rows, replace=False)
        else:
            multiply_factor = int(ratio)
            join_key_values = np.repeat(join_key_values, multiply_factor)#[:num_rows]

    feature_columns = np.random.rand(num_rows, num_columns)
    data = pd.DataFrame(feature_columns, columns=[f"feature_{i}" for i in range(1, num_columns + 1)])
    data.insert(0, join_key, join_key_values)
    return data

# Main function to generate and save the dataset
def generate_synthetic_dataset(data_id="10_2", max_columns=1500, max_single_columns=150, num_tables=10, base_rows=1000, ratios=[0.6, 0.6, 15, 25, 2, 4, 0.8, 3, 10]):
    output_dir = "synthetic_dataset_" + data_id
    total_columns = 0
    tables = []
    #base_rows = random.randint(5000, 10000)  # Number of rows in the first table

    plans_array = []
    join_serial = 0
    prev_join_id = None
    left_rows = 0
    left_cols = 0
    projection = []

    # Generate tables with size controls
    for i in range(num_tables):
        remaining_columns = max_columns - total_columns - (num_tables - len(tables) - 1)
        if remaining_columns <= 0:
            break

        num_columns = random.randint(10, min(max_single_columns, remaining_columns))  # Reduce max columns per table
        total_columns += num_columns

        table_idx = i + 1
        join_key = "join_key" + str(table_idx)

        if i == 0:
            num_rows = base_rows
            table = generate_table_optimized(table_idx, join_key, num_rows, num_columns)
            left_rows = num_rows
            left_cols = num_columns
        else:
            #k = random.uniform(0.1, 0.5)  # Reduce row growth factor
            k = ratios[i - 1]
            num_rows = int(len(tables[-1]) * k)
            #if num_rows > 100000:
            #    num_rows = 100000
            prev_join_key = "join_key" + str(i)
            table = generate_table_optimized(table_idx, join_key, num_rows, num_columns, ratio=k, join_key_values=tables[-1][prev_join_key].values)

            right_table_name = "table_" + str(i + 1)
            if join_serial == 0:
                left_table_name = "table_" + str(i)
                left_table_actual = left_table_name
            else:
                left_table_name = prev_join_id
                left_table_actual = "table_" + str(i)
            join_id = str(join_serial)

            if len(projection) == 0:
                projection.append(prev_join_key)
                projection.append("f_" + left_table_name)

            projection.append(join_key)
            projection.append("f_" + right_table_name)

            join_info = get_join_statistics(join_id, left_table_name, right_table_name, left_table_actual, prev_join_key, join_key, projection.copy(), left_rows, left_cols, num_rows, num_columns)
            plans_array.append(join_info)

            left_rows = num_rows
            left_cols += num_columns
            prev_join_id = join_id
            join_serial += 1

        tables.append(table)

    # Save tables to CSV
    os.makedirs(output_dir, exist_ok=True)
    for idx, table in enumerate(tables, start=1):
        table.to_csv(os.path.join(output_dir, f"table_{idx}.csv"), index=False)
    print(f"Dataset generated and saved in {output_dir}")

    # Save the JSON array to a file
    #with open(f"plans/{data_id}.json", "w") as json_file:
    #    json.dump(plans_array, json_file, indent=4)

    # Convert each JSON object in the array to a compact one-line format
    formatted_objects = [json.dumps(obj, separators=(',', ':')) for obj in plans_array]
    # Combine the formatted objects into a single JSON array
    formatted_json_array = "[\n\t" + ",\n\t".join(formatted_objects) + "\n]"
    # Wrap the formatted array in R"( ... )"
    formatted_string = f'R"({formatted_json_array})"'
    # Print the result
    print(formatted_string)
    # Save to a file
    with open(f"plans/{data_id}.txt", "w") as file:
        file.write(formatted_string)


# Run the script
if __name__ == "__main__":
    generate_synthetic_dataset()
