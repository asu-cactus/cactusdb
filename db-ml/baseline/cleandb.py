import utils


def delete_tables():
    db_connection = utils.get_psycopg2_connection()
    cursor = db_connection.cursor()

    # Fetch all table names
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    table_names = cursor.fetchall()
    ffnn_table_names = [table_name[0] for table_name in table_names if table_name[0].startswith("ffnn")]
    if len(ffnn_table_names) == 0:
        return 
    print("[IMPORTANT] The following tables will be deleted: ", ffnn_table_names)
    for table_name in ffnn_table_names:
        print("\t ", table_name)

    # Check if the user wants to delete tables
    confirmation = input("Are you sure you want to delete tables starting with 'ffnn'? (y/n): ")
    if confirmation.lower() == "y":
        for table_name in ffnn_table_names:
            cursor.execute("DROP TABLE IF EXISTS {}".format(table_name))
            print("Table '{}' deleted".format(table_name))
        db_connection.commit()
        print("Tables starting with 'FFNN' deleted successfully.")
    else:
        print("Nothing to do.")

if __name__ == "__main__":
    delete_tables()