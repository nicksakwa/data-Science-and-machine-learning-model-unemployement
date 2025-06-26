import numpy as np
import pandas as pd
import os
import re
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from flask import Flask, request, render_template, jsonify
import joblib # To save and load the trained model and scalers

app = Flask(__name__)

# --- 1. Load Data (for fitting encoders/scalers if not pre-saved) ---
# In a real deployment, you'd ideally save your fitted LabelEncoders and StandardScaler
# objects, as well as the trained model, to avoid re-fitting them every time the app starts.
# For simplicity, we'll refit them here, but be aware of performance implications.

# Load the original data (only needed to fit the preprocessors)
try:
    train_df = pd.read_csv("/kaggle/input/global-unemployment-data/global_unemployment_data.csv")
except FileNotFoundError:
    # If running locally, you might need to adjust the path or place the CSV in your project
    train_df = pd.read_csv("global_unemployment_data.csv") # Assume CSV is in the same directory


# --- 2. Data Preprocessing Setup (Re-fit preprocessors as done in notebook) ---
# It's crucial to use the *same* transformations on new input data as were used
# during model training.
label_encoder_country = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_age_group = LabelEncoder()
label_encoder_age_categories = LabelEncoder()

# Fit LabelEncoders on the training data's categorical columns
# We need to apply it column-wise as LabelEncoder is for single columns
train_df['country_name'] = label_encoder_country.fit_transform(train_df['country_name'])
train_df['sex'] = label_encoder_sex.fit_transform(train_df['sex'])
train_df['age_group'] = label_encoder_age_group.fit_transform(train_df['age_group'])
train_df['age_categories'] = label_encoder_age_categories.fit_transform(train_df['age_categories'])

# Drop indicator_name as it was dropped during training
train_df = train_df.drop('indicator_name', axis=1)

# Drop 2024 column and any rows with NaN in 2024 (as done in training)
train_df_cleaned = train_df.dropna(subset=['2024'])
X_train_pre_scaled = train_df_cleaned.drop('2024', axis=1)
Y_train = train_df_cleaned['2024']

# Initialize and fit the StandardScaler
scaler = StandardScaler()
scaler.fit(X_train_pre_scaled) # Fit scaler on the features used for training


# --- 3. Model Loading/Training (Retrain or Load Pre-trained) ---
# For a production application, you would typically save and load your model
# and scalers. For this example, we'll re-train the model upon app startup.
# This ensures consistency with the preprocessing.

# Create the training data using the preprocessed and scaled features
X_train_scaled = pd.DataFrame(scaler.transform(X_train_pre_scaled), columns=X_train_pre_scaled.columns)

model = HistGradientBoostingRegressor(random_state=42) # Add random_state for reproducibility
model.fit(X_train_scaled, Y_train)

# --- Define the inverse transform functions for categorical columns if needed for display ---
# For this specific model, we're predicting a numerical value (unemployment rate),
# so we don't need inverse transforms for the output.
# However, if you wanted to display the original categorical inputs, you'd use:
# label_encoder_country.inverse_transform([encoded_value]) etc.


@app.route('/')
def home():
    """Renders the home page with the prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    if request.method == 'POST':
        try:
            # Get data from the form
            country_name_raw = request.form['country_name']
            sex_raw = request.form['sex']
            age_group_raw = request.form['age_group']
            age_categories_raw = request.form['age_categories']
            year_2015 = float(request.form['2015'])
            year_2016 = float(request.form['2016'])
            year_2017 = float(request.form['2017'])
            year_2018 = float(request.form['2018'])
            year_2019 = float(request.form['2019'])
            year_2020 = float(request.form['2020'])
            year_2021 = float(request.form['2021'])
            year_2022 = float(request.form['2022'])
            year_2023 = float(request.form['2023'])
            year_2024 = float(request.form['2024'])

            # Preprocess the input data (same steps as training)
            # Ensure the order of columns matches X_train_pre_scaled
            
            # Use .transform() not .fit_transform() for new data
            # Handle potential ValueError if input label not seen during fitting
            try:
                country_name_encoded = label_encoder_country.transform([country_name_raw])[0]
            except ValueError:
                country_name_encoded = -1 # Or handle as an unknown category
                print(f"Warning: Country '{country_name_raw}' not seen in training data.")

            try:
                sex_encoded = label_encoder_sex.transform([sex_raw])[0]
            except ValueError:
                sex_encoded = -1
                print(f"Warning: Sex '{sex_raw}' not seen in training data.")

            try:
                age_group_encoded = label_encoder_age_group.transform([age_group_raw])[0]
            except ValueError:
                age_group_encoded = -1
                print(f"Warning: Age Group '{age_group_raw}' not seen in training data.")

            try:
                age_categories_encoded = label_encoder_age_categories.transform([age_categories_raw])[0]
            except ValueError:
                age_categories_encoded = -1
                print(f"Warning: Age Category '{age_categories_raw}' not seen in training data.")


            # Create a DataFrame for the single prediction, maintaining column order
            # The column names MUST exactly match those used in X_train_pre_scaled
            input_data = pd.DataFrame([[
                country_name_encoded,
                sex_encoded,
                age_group_encoded,
                age_categories_encoded,
                year_2015,
                year_2016,
                year_2017,
                year_2018,
                year_2019,
                year_2020,
                year_2021,
                year_2022,
                year_2023,
                year_2024,
            ]], columns=X_train_pre_scaled.columns) # Use columns from pre-scaled training data


            # Scale the input data using the *fitted* scaler
            scaled_input_data = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(scaled_input_data)[0]

            return render_template('index.html', prediction_text=f'Predicted 2025 Unemployment Rate: {prediction:.2f}%')

        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    # To run locally:
    # 1. Make sure global_unemployment_data.csv is in the same directory as app.py
    # 2. Open your terminal in that directory
    # 3. Run: python app.py
    # 4. Open your browser to http://127.0.0.1:5000/
    app.run(debug=True) # debug=True allows for automatic reloading on code changes