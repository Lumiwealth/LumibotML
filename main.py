from datetime import datetime
from time import time

import pandas as pd
from lumibot.backtesting import PandasDataBacktesting, YahooDataBacktesting
from lumibot.brokers import Alpaca, InteractiveBrokers
from lumibot.data_sources import PandasData
from lumibot.entities import Asset, Data
from lumibot.traders import Trader

from credentials import AlpacaConfig
from ml_strategy import MachineLearning

####
# Choose your budget and log file locations
####

strategy_name = "MachineLearning"
budget = 50000

####
# Get and Organize Data
####

symbol = "SRNE"
asset = Asset(symbol=symbol, asset_type="stock")
df = pd.read_csv(f"data/{asset}_Minute.csv")
df = df.set_index("time")
df.index = pd.to_datetime(df.index)
# print(df)
backtesting_start = datetime(2021, 8, 1)
backtesting_end = datetime(2021, 8, 3)

my_data = dict()
my_data[asset] = Data(
    asset=asset,
    df=df,
    timestep="minute",
)

####
# Backtest
####

pandas = PandasData(my_data)

MachineLearning.backtest(
    PandasDataBacktesting,
    backtesting_start,
    backtesting_end,
    config=AlpacaConfig,
    pandas_data=my_data,
    symbol=asset,
    name=strategy_name,
    budget=budget,
)

####
# Run the strategy
####

ac = AlpacaConfig()
# APCA_API_KEY_ID = ac.API_KEY
# APCA_API_SECRET_KEY = ac.API_SECRET

broker = Alpaca(ac)
strategy = MachineLearning(
    name=strategy_name,
    budget=budget,
    broker=broker,
)
trader = Trader()
trader.add_strategy(strategy)
trader.run_all()
