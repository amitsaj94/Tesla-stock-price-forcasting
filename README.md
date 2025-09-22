
# Tesla Stock Price Forecasting Using Prophet

Welcome to this project that demonstrates forecasting Tesla’s daily adjusted close stock price using time series techniques, feature engineering, and the Facebook Prophet library.

---

## Project Overview

- **Goal:** Predict Tesla stock prices leveraging historical data with lagged and rolling statistical features.  
- **Model:** Facebook Prophet enhanced with custom regressors for improved accuracy.  
- **Data:** Includes daily Open, High, Low, Close, Adjusted Close prices, and Trading Volume.  
- **Techniques:** Log transformation, lag features, rolling means/standard deviations, and evaluation with RMSE and MAE.

---

## Data Exploration & Visualization

- Time series plots reveal Tesla’s stock price trends and volume fluctuations over time.  
- Correlation heatmaps show strong positive relationships among price metrics, and volume exhibits more complex patterning.  
- Feature engineering incorporates recent trend and volatility indicators through lagged and rolling statistics.
- Candlestick Chart with Volume offers deeper insight into daily price movements and market sentiment compared to simple line charts.
The green/red candles indicate days with price increases or decreases, and long wicks highlight volatility and price range for each day.
- Seasonal Decomposition helps uncover underlying patterns: long-term trend direction, recurring seasonal fluctuations (e.g., calendar effects), and noise, which informs feature engineering and model choice.

---

## Model Preparation & Forecasting

- The dataset is split into training (all but last 365 days) and test sets (last 365 days).  
- The Prophet model is initialized with daily and yearly seasonalities.  
- Custom regressors (`y_lag1`, `y_lag2`, `rolling_mean_4d`) representing recent price behavior are added.  
- Model fitting is performed on training data, with forecasting on the test set to assess predictive power.

---

## Evaluation Metrics

- Forecasted values are transformed back from the log scale for interpretation.  
- Performance is evaluated using:  
  - **Root Mean Squared Error (RMSE)**  
  - **Mean Absolute Error (MAE)**  
- These provide numerical measures of forecast accuracy on original price scale.

# Insights on Metrics

- **Root Mean Squared Error (RMSE) = 30.20**  
- **Mean Absolute Error (MAE) = 21.25**
  
The model's performance metrics indicate a strong forecast. The RMSE of 30.20 and MAE of 21.25 show that, on average, the model's predictions are off by about $21 on a stock that trades for hundreds of dollars. This is a very small error relative to the stock's price, demonstrating high accuracy

---

## Results Visualization

<img width="1458" height="610" alt="Screenshot 2025-09-22 121317" src="https://github.com/user-attachments/assets/4115e01e-8eea-4d13-88d4-7b6ba36835fa" />


- **Blue Line:** Actual Tesla adjusted close prices.  
- **Red Line:** Prophet forecasted prices.  

The close alignment demonstrates the model's effectiveness in capturing price movements and market trends. Some deviations during volatile periods highlight areas for potential enhancement.

---

## Getting Started

### Prerequisites

- Python 3.8+  
- Required libraries listed in `requirements.txt`

### Installation

Clone the repo:

git clone https://github.com/yourusername/tesla-stock-forecast.git

text

Install dependencies:

pip install -r requirements.txt

text

Run the notebook:

- Open `notebooks/tesla_prophet_forecast.ipynb` in Jupyter or Google Colab.  
- Follow the documented steps for data processing, model training, prediction, and evaluation.

---

## Future Work

- Incorporate additional regressors such as market indices or sentiment.  
- Test alternative forecasting methods like LSTM or XGBoost.  
- Automate hyperparameter tuning and model validation.

---

## Author

Amit Sajwan

---

Feel free to reach out with feedback or contributions!
