#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline


def load_data():

    data = 'https://raw.githubusercontent.com/aliaksandra-babova/ML-Zoomcamp-2025-Midterm-Project/refs/heads/main/play_tennis_dataset.csv'

    df = pd.read_csv('play_tennis_dataset.csv')

    df = df.drop(columns = ['Day'])

    df.columns = df.columns.str.lower()

    categorical = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical:
        df[c] = df[c].str.lower()

    df.outlook = df.outlook.fillna("unknown")
    df.temperature = df.temperature.fillna("unknown")
    df.humidity = df.humidity.fillna("unknown")
    df.wind = df.wind.fillna("unknown")

    df.play = (df.play == 'no').astype(int)

    return df

def train_model(df):

    pipeline = make_pipeline(
    DictVectorizer(),
    LogisticRegression(solver='liblinear')
    )

    categorical = list(df.dtypes[df.dtypes == 'object'].index)

    train_dict = df[categorical].to_dict(orient='records')
    y_train = df.play.values
    pipeline.fit(train_dict, y_train)

    return pipeline

def save_model(filename, model):

    with open(filename, 'wb') as f_out: 
        pickle.dump(model, f_out)

    print(f'Model saved to {filename}')

df = load_data()
pipeline = train_model(df)
save_model('model.bin', pipeline)





