import logging
from datetime import datetime, timedelta
from time import time

import pandas as pd
from lumibot.backtesting import (
    AlpacaBacktesting,
    AlphaVantageBacktesting,
    BacktestingBroker,
    PandasDataBacktesting,
    YahooDataBacktesting,
)
from lumibot.brokers import Alpaca, InteractiveBrokers
from lumibot.data_sources import PandasData
from lumibot.tools import indicators
from lumibot.traders import Trader
from twilio.rest import Client

from credentials import AlpacaConfig
from ml_strategy import MachineLearning

# Choose your budget and log file locations
budget = 50000
logfile = "logs/test.log"
benchmark_asset = None

# Initialize all our classes
trader = Trader(logfile=logfile)

# Development: Minute Data
asset = "SRNE"
df = pd.read_csv(f"data/{asset}_Minute.csv")
df = df.set_index("time")
df.index = pd.to_datetime(df.index)
my_data = dict()
my_data[asset] = df
backtesting_start = datetime(2021, 8, 1)
backtesting_end = datetime(2021, 9, 1)

####
# Select our strategy
####

pandas = PandasData(my_data)
broker = BacktestingBroker(pandas)

strategy_name = "MachineLearning"

####
# Backtest
####

MachineLearning.backtest(
    strategy_name,
    budget,
    PandasDataBacktesting,
    backtesting_start,
    backtesting_end,
    config=AlpacaConfig,
    pandas_data=my_data,
    symbol=asset,
)

####
# Run the strategy
####

broker = Alpaca(AlpacaConfig)
strategy = MachineLearning(
    name=strategy_name,
    budget=budget,
    broker=broker,
)
trader.add_strategy(strategy)
trader.run_all()
