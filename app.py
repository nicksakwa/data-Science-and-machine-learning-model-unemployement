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

train_df = train.df.drop('indicator_name', axis=1)
train_df_cleaned = train.df.dropna(subnet=['2024'])
X_train_pre_scaled = train.df_cleaned.drop('2024', axis=1)
Y_train= train.df_cleaned['2024']

scaler=StandardScaler()
scaler.fit(X_train_pre_scaled)
X_trained_scaled=pd.DataFrame(scaler.transform(X_trained_pre_scaled),columns=X_trained_scaled.columns)
model=HistGradientBoostingRegressor(random_state=42)
model.fit(X_train_scaled, Y_train)

@app.route('/')
def home():
    """Renders from the home page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction request."""
    if request.method == 'POST':
        try:
            country_name_raw = request.form['country_name']
            sex_raw = request.form['sex']
            age_group_raw = request.form['age_group']
            age_categories_raw = ['age_categories']
            year_2014 = float(request.form['2014'])
            year_2015 = float(request.form['2015'])
            year_2016 = float(request.form['2016'])
            year_2017 = float(request.form['2017'])
            year_2018 = float(request.form['2018'])
            year_2019 = float(request.form['2019'])
            year_2020 = float(request.form['2020'])
            year_2021 = float(request.form['2021'])
            year_2022 = float(request.form['2022'])
            year_2023 = float(request.form['2023'])
            
            try:
                country_name_encoded= label_encoder_country.transform([country_name])[0]
            except valueError:
                country_name_encoded= -1
                print(f"Warning: country '{country_name_raw}'not seen in training data.")

