from gcp_client import GcpClient
from google.cloud import bigquery, storage
import pandas as pd
from pathlib import Path

SAMPLE_SPEC = {"max_rows": 10000,
               "tablesample_pct": 0.1}

def bigquery_query_runner(query, gcp_client):
    """
    Args:
        query (str): BigQuery query string
        gcp_client (GcpClient): GcpClient object
    """
    query_job = gcp_client.bq_client.query(query)
    return query_job.to_dataframe()

class Table:
    """
    types:
      - bigquery.TableReference: BigQuery table (support '*' globbing table name)
      - pandas.DataFrame
      - Path: Local CSV file (support '*' globbing file name)
      - storage.Blob: GCS CSV file (support '*' globbing file name)
    """
    def __init__(self, gcp_client, table_obj, sample_spec=SAMPLE_SPEC, col_subset: list[str] = None):
        _table_types = {
          bigquery.TableReference: "bigquery",
          pd.DataFrame: ("local", "pandas_dataframe"),
          Path: ("local", "csv"),
          storage.Blob: ("remote", "gcs_csv")
        }
        
        self.gcp_client = gcp_client
        self._table_obj = table_obj
        self.storage_loc, self.table_type = _table_types[type(table_obj)]
        
        self.sample_spec = sample_spec
      
    def _df_col_subset(df, col_subset):
        return df[col_subset] if col_subset is not None else df

    def read_table(self):
        if isinstance(self._table_obj, bigquery.TableReference):
            query = f"SELECT * FROM `{self._table_obj.project}.{self._table_obj.dataset_id}.{self._table_obj.table_id}`"
            if SAMPLE_SPEC['tablesample_pct'] is not None:
                query += f"\nTABLESAMPLE ({SAMPLE_SPEC['tablesample_pct']} PERCENT)"
            if SAMPLE_SPEC['max_rows'] is not None:
                query += f"\nLIMIT {SAMPLE_SPEC['max_rows']}"
            
            query_job = self.gcp_client.query(query)
            df = query_job.result().to_dataframe()
        elif isinstance(self._table_obj, pd.DataFrame):
            df = self._table_obj
        elif isinstance(self._table_obj, Path):
            df = pd.read_csv(self._table_obj)
        elif isinstance(self._table_obj, storage.Blob):
            # TODO: pretty sure GCS won't work this way
            df = pd.read_csv(self._table_obj)
        
        return df_filtered
