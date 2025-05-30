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

