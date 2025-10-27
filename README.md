
## **Electric Vehicle Charging Demand Forecasting**

## Overview

This project focuses on forecasting electric vehicle (EV) charging demand using data-driven machine learning methods. The analysis helps identify peak demand hours and station utilization patterns to support smart energy and infrastructure planning.

The entire workflow — from data preprocessing to forecasting and visualization — is implemented in a single Jupyter Notebook: `ev_station_analysis.ipynb`, ensuring simplicity, reproducibility, and clarity.


## Objectives

* Predict hourly EV charging demand at the station and city level.
* Analyze temporal (hour/day) and weather-based demand patterns.
* Generate actionable insights for optimizing charging station distribution.
* Visualize results interactively using Tableau and Python plots.


## Key Features

 Complete analysis implemented in Jupyter Notebook (`ev_station_analysis.ipynb`)
-> Data preprocessing, feature engineering, and weather data integration
-> XGBoost-based regression forecasting for hourly and city-level demand
-> Evaluation using MAE and RMSE metrics
-> Visualization of station and time-level patterns (hour × weekday)
-> Optional Streamlit dashboard for browser-based interaction


## Tools & Technologies
                                                               
Tool         -Purpose
Python       -(Prophet, XGBoost)	Data cleaning, forecasting, evaluation
Excel	       -Data merging and initial analysis
Tableau      -Dashboard & visualization
Matplotlib / Pandas- Visualization and analysis
Environment	- Jupyter Notebook


## Modeling Workflow

1 Data Preparation

* Aggregated EV charging session data into hourly demand per station.
* Merged weather data (temperature, humidity, etc.) using timestamps.
* Created new features like hour of day, day of week, and encoded station IDs.

2 Station-Level Analysis (Jupyter Notebook: `ev_station_analysis.ipynb`)

The full project workflow — including preprocessing, model training, and visualization — is implemented in this notebook.
It:

* Performs exploratory data analysis (EDA) on station-level usage.
* Trains an XGBoost regression model for forecasting hourly demand.
* Evaluates performance using MAE and RMSE.
* Produces final forecasts and visual insights for each station.

> *(Note: The Python script `ev_forecast_train.py` provides a command-line version of the training process. Running it is optional and not required for reproducing results.)*

3 Evaluation Metrics
| Model   | MAE   | RMSE  |
| XGBoost | 2.513 | 4.668 |

## Visualization

The results were visualized using Tableau and Python-based charts:

* Demand Heatmap: Hour vs. Weekday — to highlight daily patterns.
* City-Level Forecast: Displays overall predicted demand trends.
* Station-Level Plots: Show localized demand fluctuations for individual stations.


## Results & Insights

* Peak charging demand occurs between 18:00–21:00 hours.
* Weekdays show consistently higher utilization compared to weekends.
* The XGBoost model achieved strong performance (MAE: 2.513, RMSE: 4.668).
* Station-level analysis revealed key hotspots for future charger placement.


## Project Folder Structure

```
Electric-Vehicle-Charging-Demand-Forecasting/
│
├── data/                         # Raw and processed datasets
│   ├── ev_sessions.csv           # EV charging session data
│   ├── weather_hourly.csv        # Weather dataset
│   ├── ev_city_hist.xls          # Historical aggregated demand data
│   └── ev_city_forecast_xgb.xls  # Forecasted city-level demand (XGBoost)
│
├── notebooks/                    # Main notebook for analysis
│   └── ev_station_analysis.ipynb # End-to-end station-level analysis and forecasting
│
├── scripts/                      # Optional command-line scripts
│   ├── ev_forecast_train.py      # Optional: model training via command line
│   └── streamlit_dashboard.py    # Optional: Streamlit dashboard for visualization
│
├──visuals/                       #Output visualizations
│   ├── feature importance.png
│   └── Forecast line chart.png
│   └──forecast plot.png
│   └──Heatmap.png
|
├── outputs/                      # Generated models and metrics
│   ├── ev_pooled_xgb_model.json  # Saved XGBoost model
│   └── metrics.csv               # Evaluation results (MAE, RMSE)
│
├── Electric Vehicle Charging Demand Forecasting.docx  #contains detailed report
├── EV_Demand_Dashboard.twb       # Visualization
├── README.md                     # Project overview and documentation
└── requirements.txt              # Python dependencies
```



## Future Improvements

* Integrate Prophet for trend-based forecasting comparison.
* Expand per-station forecasting automation.
* Deploy Streamlit app or Tableau public dashboard.
* Incorporate real-time data streaming for continuous updates.


## Conclusion

This project successfully demonstrates the forecasting of EV charging demand using machine learning and time-based analysis.
The implementation emphasizes clarity, reproducibility, and interpretability — key principles expected in professional data science workflows.

