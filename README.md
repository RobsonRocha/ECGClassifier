# ECGClassifier

Third project from [Machine Learning Engineer Nanodegree Program Udacity course](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t)

## Motivation

This project consists in predict if an Electrocardiogram (ECG) presents some disease or not. To reach this goal, it will be compared two algorithms' answers and it will be check if it is possible to detect a healthy (or not) signal, just understanding its measures.
It will be show how to use AWS Sagemaker features to do that.
It contains a python notebook that has feature engeneering, training, testing and validation and it shows how to tuning XGboost hyperparameters. Also it contains and a study about choose the top 5 relevant features to both algorithms using random forest algorithm.
XBoost's performance will be faced with CatBoost Classifier.

## Training and testing

All process to get the data file, transform it, training, testing and create the endpoint and use it is explain in notebook file [ECGClassifier.ipynb](./ECGClassifier.ipynb).

The notebook also shows two ways of how to training, testing and deploy. To XGBoost it was used [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) library and to CatBoost Classifier, it was used [scikit learn](https://scikit-learn.org/stable/) framework.
