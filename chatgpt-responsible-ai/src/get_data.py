from google.cloud import bigquery
import pandas as pd

def read_from_bigquery(project_id, dataset_id, table_id):
    # Initialize a BigQuery client
    client = bigquery.Client(project=project_id)
    
    # Construct a reference to the table
    table_ref = client.dataset(dataset_id).table(table_id)

    # Read the data from BigQuery into a Pandas DataFrame
    df = client.query(f"SELECT * FROM `{table_ref}`").to_dataframe()

    return df