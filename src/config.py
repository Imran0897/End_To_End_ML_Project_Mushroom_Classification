from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd
import json
from dataclasses import dataclass
# Provide the mongodb localhost url to connect python to mongodb.
import os
import pandas as pd
from src.logger import logging


@dataclass
class EnvironmentVariable:
    client_id:str = os.getenv("CLIENT_ID")
    client_secret:str = os.getenv("CLIENT_SECRET")
    secured_connect_path:str = os.getenv('SECURED_CONNECT_PATH')
    key_space:str = os.getenv("KEY_SPACE")

TARGET_COLUMN_MAPPING = {"p":1,"e":0}

env_var = EnvironmentVariable()
cloud_config = { 'secure_connect_bundle' : env_var.secured_connect_path}
auth_provider = PlainTextAuthProvider(env_var.client_id, env_var.client_secret)
cassandra_client = Cluster(cloud=cloud_config, auth_provider=auth_provider)

# mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
TARGET_COLUMN = "class"

