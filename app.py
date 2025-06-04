import numpy as np
import pandas as pd
import os
import re
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HisGradientBoostingRegressor
from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

try:
    train_df=pd.read_csv("/kaggle/input/global-unemployment-data/global_unemployment_data.csv")
except FileNotFoundError:
    train_df=pd.read_csv("global_unemployement_data.csv")

label_encoder_country= LabelEncoder()
label_encoder_sex= LabelEncoder()
label_encoder_age_group= LabelEncoder()
label_encoder_age_categories=LabelEncoder()

train_df['country_name']=label_encoder_country.fit_transform(train_df['country_name'])
train_df['sex']=label_encoder_sex.fit_transform(train_df['sex'])
train_df['age_group']=label_encoder_age_group.fit_transform(train_df['age_group'])
train_df['age_categories']=label_encoder_age_categories.fit_transform(train_df['age_categories'])
