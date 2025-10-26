# Streamlit example (streamlit_app.py)
import streamlit as st
import pandas as pd
import plotly.express as px

city_forecast = pd.read_csv('../data/ev_city_forecast_xgb.xls', parse_dates=['ds'])

fig_line = px.line(city_forecast, x='ds', y='city_pred_xgb', title="City EV Forecast")
fig_heat = px.density_heatmap(city_forecast, x=city_forecast['ds'].dt.hour, 
                              y=city_forecast['ds'].dt.dayofweek,
                              z='city_pred_xgb', 
                              color_continuous_scale='YlOrRd',
                              labels={'x':'Hour','y':'Day of Week','z':'Predicted kWh'})

st.title("EV Charging Demand Dashboard")
st.plotly_chart(fig_line)
st.plotly_chart(fig_heat)
