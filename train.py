from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.model_selection import train_test_split

# Create TabularDataset using TabularDatasetFactory
# Data is available at: 
# "https://datahub.io/machine-learning/diabetes/r/diabetes.csv"

### YOUR CODE HERE ###
URL = 'https://datahub.io/machine-learning/diabetes/r/diabetes.csv'
ds = TabularDatasetFactory.from_delimited_files(path=URL)
# Use the clean_data function to clean your data.
x = ds.to_pandas_dataframe().dropna() 
y = x.pop("class").apply(lambda s: 1 if s == "tested_positive" else 0)
x_train ,x_test ,y_train ,y_test = train_test_split(x, y ,test_size=0.8,random_state=42,shuffle=True)
#train_data = x_train * y_train
#train_data, valid_data = ds.random_split (percentage=0.8,seed=0)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    AUC_weighted = model.score(x_test, y_test)
    run.log("AUC_weighted", np.float(AUC_weighted))

if __name__ == '__main__':
    main()
