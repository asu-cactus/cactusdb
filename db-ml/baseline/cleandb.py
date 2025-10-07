import utils


def delete_postgres_tables():
    db_connection = utils.get_psycopg2_connection()
    cursor = db_connection.cursor()

    # Fetch all table names
    cursor.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    )
    table_names = cursor.fetchall()
    ffnn_table_names = [
        table_name[0] for table_name in table_names if table_name[0].startswith("ffnn")
    ]
    if len(ffnn_table_names) == 0:
        return
    print("[IMPORTANT] The following tables will be deleted: ")
    for table_name in ffnn_table_names:
        print("\t ", table_name)

    # Check if the user wants to delete tables
    confirmation = input(
        "[Postgres] Are you sure you want to delete tables starting with 'ffnn'? (y/n): "
    )
    if confirmation.lower() == "y":
        for table_name in ffnn_table_names:
            cursor.execute("DROP TABLE IF EXISTS {}".format(table_name))
            print("Table '{}' deleted".format(table_name))
        db_connection.commit()
        print("Tables starting with 'FFNN' deleted successfully.")
    else:
        print("Nothing to do.")


def delete_hdfs_data():
    result = utils.ls_hdfs_dir("/user/velox/data/")
    if result.returncode == 0:
        # Parse the output to extract filenames (assuming basic format)
        output_lines = result.stdout.decode().strip().splitlines()
    else:
        return
    if len(output_lines) == 0:
        return
    ffnn_table_names = [
        table_name for table_name in output_lines if "ffnn_data" in table_name
    ]

    print("[IMPORTANT] The following tables will be deleted: ")
    for table_name in ffnn_table_names:
        print("\t ", table_name)

    # Check if the user wants to delete tables
    confirmation = input(
        "[HDFS] Are you sure you want to delete tables starting with 'ffnn'? (y/n): "
    )
    if confirmation.lower() == "y":
        for table_name in ffnn_table_names:
            utils.rm_hdfs_file(table_name)
            print("File '{}' deleted".format(table_name))
        print("File starting with 'FFNN' deleted successfully.")
    else:
        print("Nothing to do.")


def drop_all_views():
    utils.execute_sql_query_via_psycopg2(
        """
    DO $$ 
    DECLARE
        view_name RECORD;
    BEGIN
        FOR view_name IN 
            SELECT table_schema, table_name 
            FROM information_schema.views 
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema') 
        LOOP
            EXECUTE format('DROP VIEW IF EXISTS %I.%I CASCADE;', view_name.table_schema, view_name.table_name);
        END LOOP;
    END $$;
                                         """
    )


if __name__ == "__main__":
    delete_postgres_tables()
    delete_hdfs_data()
    drop_all_views()
