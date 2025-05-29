from google.cloud.bigquery import Client as bq_client
from google.cloud.storage import Client as gcs_client

class GcpClient:
    def __init__(self, project_id):
        self.project_id = project_id
        self.bq_client = bq_client(project_id)
        self.gcs_client = gcs_client(project_id)
