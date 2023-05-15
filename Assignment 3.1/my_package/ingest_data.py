#!/usr/bin/env python
# coding: utf-8


# import library
import os
import tarfile
import urllib
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--DOWNLOAD_ROOT",help="give download data path",default='' )
args = parser.parse_args()


def fetch_housing_data(HOUSING_PATH,housing_url=HOUSING_URL):
    os.makedirs(HOUSING_PATH, exist_ok=True)
    tgz_path = os.path.join(HOUSING_PATH, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=HOUSING_PATH)
    housing_tgz.close()

# create pandas dataframe
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5000
# remote_server_uri = "http://127.0.0.1:5000" # set to your server URI
# mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

# exp_name = "House_price_prediction_data_spliting"
# mlflow.set_experiment(exp_name)

def ingest_data(download_root):
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_PATH = os.path.join(download_root,HOUSING_PATH)
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    #mlflow.log_param("output path", HOUSING_PATH)
    fetch_housing_data(HOUSING_PATH)
    housing=load_housing_data()
    # method 4
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    train_set.to_csv(HOUSING_PATH+"//train.csv")
    test_set.to_csv(HOUSING_PATH+"//validate.csv")
    return

ingest_data(args.DOWNLOAD_ROOT)
