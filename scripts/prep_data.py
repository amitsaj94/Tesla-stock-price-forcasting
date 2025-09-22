import argparse, pandas as pd, numpy as np

def build_features(df):
    df = df.sort_values('Date').copy()
    df['y_orig'] = df['Adj Close']
    df['y'] = np.log(df['Adj Close'])
    df['y_lag1'] = df['y'].shift(1)
    df['y_lag2'] = df['y'].shift(2)
    df['rolling_mean_4d'] = df['y'].rolling(window=4).mean().shift(1)
    return df.dropna()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input, parse_dates=['Date'])
    out = build_features(df)
    out.to_csv(args.output, index=False)
