#!/usr/bin/env python
# coding: utf-8


import os
import tarfile
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def score(validate_file_path='',pickle_file_path=''):
    
    HOUSING_PATH = os.path.join("datasets", "housing","validate.csv")
    HOUSING_PATH = os.path.join(validate_file_path,HOUSING_PATH)
    #HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    #mlflow.log_param("output path", HOUSING_PATH)
    
    housing = pd.read_csv(HOUSING_PATH)
    housing["income_cat"] = pd.cut(housing["median_income"],
                           bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                           labels=[1, 2, 3, 4, 5])
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"]=housing["population"]/housing["households"]
    housing1=housing
    housing = housing1.drop("median_house_value", axis=1)
    housing_labels = housing1["median_house_value"].copy()
    housing.drop("income_cat", axis=1,inplace=True)
    #mlflow.log_param("data_shape", housing.shape)

    median = housing["total_bedrooms"].median()  # option 3
    housing["total_bedrooms"].fillna(median, inplace=True)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    imputer.statistics_
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                      index=housing_num.index)
    housing_cat = housing[["ocean_proximity"]]
    # OrdinalEncoder
    from sklearn.preprocessing import OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    from sklearn.preprocessing import OneHotEncoder
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    housing_cat_1hot
    housing_cat_1hot.shape

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])

    housing_num_tr = num_pipeline.fit_transform(housing_num)

    from sklearn.compose import ColumnTransformer

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)
    #mlflow.log_param("data_shape", housing_prepared.shape)

    # linear Regression
    from sklearn.linear_model import LinearRegression
    filename = os.path.join(pickle_file_path, 'linear.pkl')#args.output_model_path
    #mlflow.log_param('input_pickle_file',filename)
    from joblib import Parallel, delayed
    import joblib
    lr_from_joblib = joblib.load(filename)
    prediction = lr_from_joblib.predict(housing_prepared)   
    import math
    import sklearn.metrics
    mse = sklearn.metrics.mean_squared_error(housing_labels, prediction)
    rmse = math.sqrt(mse)
    #mlflow.log_metric('rmse',rmse)
    print(rmse)
    return

#score(args.read_validate_file,args.read_model_path)