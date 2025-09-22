import argparse, joblib, pandas as pd
from prophet import Prophet

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.data, parse_dates=['Date']).rename(columns={'Date':'ds','y':'y'})
    REGRESSORS = ['y_lag1','y_lag2','rolling_mean_4d']

    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    for r in REGRESSORS:
        model.add_regressor(r)
    model.fit(df[:-365])  # or train/test split externally
    joblib.dump(model, args.out)
