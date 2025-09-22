import argparse, joblib, pandas as pd, numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    args = p.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.data, parse_dates=['ds'])
    test = df[-365:].reset_index(drop=True)
    future = test[['ds','y_lag1','y_lag2','rolling_mean_4d']]
    fc = model.predict(future)
    test['y_pred'] = np.exp(fc['yhat'])
    test['y_actual'] = np.exp(test['y'])
    rmse = np.sqrt(mean_squared_error(test['y_actual'], test['y_pred']))
    mae = mean_absolute_error(test['y_actual'], test['y_pred'])
    print("RMSE:", rmse, "MAE:", mae)
    plt.figure(figsize=(12,6)); plt.plot(test['ds'], test['y_actual'], label='Actual'); plt.plot(test['ds'], test['y_pred'], label='Forecast')
    plt.savefig("reports/figures/forecast_vs_actual.png", bbox_inches='tight')
