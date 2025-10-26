#!/usr/bin/env python3
"""
ev_forecast_train.py
Run the per-station EV forecasting pipeline (Prophet optional + pooled XGBoost)
Usage:
    python ev_forecast_train.py --data_dir ../data --out_dir ../outputs --use_prophet
"""
import argparse, os
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


def run(data_dir, out_dir, use_prophet=False):
    ev_path = os.path.join(data_dir, 'ev_sessions.csv')
    weather_path = os.path.join(data_dir, 'weather_hourly.csv')
    df = pd.read_csv(ev_path, parse_dates=['start_time'])
    weather = pd.read_csv(weather_path, parse_dates=['ds'])
    df = df.dropna(subset=['start_time'])
    df = df[df['energy_kwh']>0]
    df['ds'] = df['start_time'].dt.floor('H')
    hourly_station = df.groupby(['station_id','ds']).agg(energy_kwh=('energy_kwh','sum'), sessions=('session_id','count')).reset_index()
    stations = hourly_station['station_id'].unique()
    full_idx = pd.date_range(start=hourly_station['ds'].min(), end=hourly_station['ds'].max(), freq='H')
    frames = []
    for st in stations:
        tmp = pd.DataFrame({'ds': full_idx}); tmp['station_id'] = st; frames.append(tmp)
    full_df = pd.concat(frames, ignore_index=True).merge(hourly_station,on=['station_id','ds'],how='left')
    full_df['energy_kwh'] = full_df['energy_kwh'].fillna(0)
    full_df['sessions'] = full_df['sessions'].fillna(0)
    full_df = full_df.merge(weather, on='ds', how='left')
    full_df['temperature'] = full_df['temperature'].fillna(full_df['temperature'].mean())
    full_df['precipitation'] = full_df['precipitation'].fillna(0)
    full_df['hour'] = full_df['ds'].dt.hour; full_df['dow'] = full_df['ds'].dt.dayofweek
    le = LabelEncoder(); full_df['station_le'] = le.fit_transform(full_df['station_id'])
    features = ['station_le','hour','dow','temperature','precipitation','sessions']
    unique_ds = np.sort(full_df['ds'].unique()); cutoff = unique_ds[int(len(unique_ds)*0.85)]
    train_mask = full_df['ds'] <= cutoff
    X_train = full_df.loc[train_mask, features]; y_train = full_df.loc[train_mask, 'energy_kwh']
    X_test = full_df.loc[~train_mask, features]; y_test = full_df.loc[~train_mask, 'energy_kwh']
    model = XGBRegressor(n_estimators=300, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds); rmse = mean_squared_error(y_test, preds, squared=False)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({'model':['pooled_xgb'],'MAE':[mae],'RMSE':[rmse]}).to_csv(os.path.join(out_dir,'metrics.csv'), index=False)
    model.save_model(os.path.join(out_dir,'ev_pooled_xgb_model.json'))
    print('Saved outputs to', out_dir)

if __name__=='__main__':
    data_dir = r"C:\Users\javid\OneDrive\Documents\Project 1"
    out_dir = r"C:\Users\javid\OneDrive\Documents\Project 1\outputs"
    run(data_dir, out_dir, use_prophet=False)

