#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import json
import pandas as pd
import numpy as np
os.system('pip install joblib==0.16.0')
import joblib
import sys
os.system('pip install imblearn')
os.system('pip install catboost')
os.system('pip install sagemaker==1.69.0')
os.system('pip install s3fs==0.4.2')
from catboost import CatBoostClassifier, Pool
import boto3
import sagemaker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyperparameters are described here
    parser.add_argument('--learning_rate', type=float, default=0.03)
    
    
    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args = parser.parse_args()   
    
    print(args.train)
    
    print(args.validation)
    
    train_data = pd.read_csv(args.train+"/train", names=["Class","bpm","ibi","sdnn","sdsd","rmssd","pnn20","pnn50","hr_mad","sd1","sd2","s","sd1/sd2","breathingrate"])
    val_data = pd.read_csv(args.validation+"/valid", names=["Class","bpm","ibi","sdnn","sdsd","rmssd","pnn20","pnn50","hr_mad","sd1","sd2","s","sd1/sd2","breathingrate"])
    
    train_X = train_data.iloc[:,1:13]
    train_y = train_data.iloc[:,0:1]
    
    print(train_X)
    print(train_y)
    
    val_X = val_data.iloc[:,1:13]
    val_y = val_data.iloc[:,0:1]
    
    print(val_X)
    print(val_y)
    
    eval_dataset = Pool(val_X,
                    val_y)

    model = CatBoostClassifier(learning_rate=args.learning_rate,
                           custom_metric=['Logloss',
                                          'AUC:hints=skip_train~false'])

    predictor = model.fit(train_X, train_y, eval_set=eval_dataset, verbose=False)

    print(model.get_best_score())    
    
    
    joblib.dump(predictor, os.path.join(args.model_dir, "model.joblib"))

def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    cbc = joblib.load(os.path.join(model_dir, "model.joblib"))
    return cbc