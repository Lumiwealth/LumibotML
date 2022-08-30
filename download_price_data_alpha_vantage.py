# Make sure to run `pip install tqdm` before this script
import time

import pandas as pd
from tqdm import tqdm

symbols = ["VIX", "VXX", "XIV", "PBP", "SPXL"]
interval = "1min"
api_key = "30WM6G3P2TVGCIWL"

years = 2
months = 12
with tqdm(total=months * years * len(symbols)) as pbar:
    for symbol in symbols:
        dfs = []
        for y in range(years):
            for m in range(months):
                slice = f"year{y+1}month{m+1}"
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={symbol}&interval={interval}&slice={slice}&apikey={api_key}"
                df = pd.read_csv(url)
                dfs.append(df)
                time.sleep(
                    13
                )  # API call limit is 5 per minute - including some buffer time here
                pbar.update(1)

        df_all = pd.concat(dfs)
        df_all.to_csv(f"data/{symbol}_{interval}.csv")
