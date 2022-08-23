from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ta
from lumibot.backtesting import PandasDataBacktesting, YahooDataBacktesting
from lumibot.brokers import Alpaca, InteractiveBrokers
from lumibot.data_sources import PandasData
from lumibot.entities import Asset, Data
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from credentials import AlpacaConfig
from ml_strategy_stocks import MachineLearning


class MachineLearning(Strategy):

    # =====Overloading lifecycle methods=============

    def initialize(
        self,
        symbol="SRNE",
        compute_frequency=15,
        lookback_period=2000,
        pct_portfolio_per_trade=0.20,
        price_change_threshold_up=0.01,
        price_change_threshold_down=0.01,
        max_pct_portfolio=1,
    ):
        """Initializes all the variables and settings for the strategy

        Args:
            symbol (str, optional): The symbol that we want to trade. Defaults to "SRNE".
            compute_frequency (int, optional): The time (in minutes) that we should retrain our model.
            lookback_period (int, optional): The amount of data (in minutes) that we get from our data source to use in the model.
            pct_portfolio_per_trade (float, optional): The size that each trade will be (in percent of the total portfolio).
            price_change_threshold_up (float, optional): The difference between predicted price and the current price that will trigger a buy order (in percentage change).
            price_change_threshold_down (float, optional): The difference between predicted price and the current price that will trigger a sell order (in percentage change).
            max_pct_portfolio (float, optional): The maximum that the strategy will buy or sell as a percentage of the portfolio (eg. if this is 0.8 - or 80% - and our portfolio is worth $100k, then we will stop buying when we own $80k worth of the symbol)
        """
        # Set the initial variables or constants

        # Built in Variables
        if self.is_backtesting:
            # If we are backtesting we do not need to check very often
            self.sleeptime = compute_frequency
        else:
            self.sleeptime = "10S"

        # Settings
        self.minutes_before_closing = 5

        self.symbol = symbol
        self.compute_frequency = compute_frequency
        self.lookback_period = lookback_period
        self.pct_portfolio_per_trade = pct_portfolio_per_trade
        self.price_change_threshold_up = price_change_threshold_up
        self.price_change_threshold_down = price_change_threshold_down
        self.max_pct_portfolio = max_pct_portfolio

        self.last_compute = None
        self.prediction = None
        self.last_price = None
        self.asset_value_potential = None
        self.shares_owned_potential = None

    def on_trading_iteration(self):
        current_datetime = self.get_datetime()
        time_since_last_compute = None
        if self.last_compute is not None:
            time_since_last_compute = current_datetime - self.last_compute

        if time_since_last_compute is None or time_since_last_compute > timedelta(
            minutes=self.compute_frequency
        ):
            self.last_compute = current_datetime

            # The current price of the asset
            self.last_price = self.get_last_price(self.symbol)

            # Get how much we currently own of the asset
            self.shares_owned_potential = self.get_asset_potential_total(self.symbol)
            self.asset_value_potential = (
                abs(self.shares_owned_potential) * self.last_price
            )

            data = self.get_data(
                self.symbol, self.compute_frequency * self.lookback_period
            )

            data["close_future"] = data["close"].shift(-self.compute_frequency)
            data_train = data.dropna()

            self.portfolio_value
            self.cash

            # Predict
            rf = RandomForestRegressor().fit(
                X=data_train.drop("close_future", axis=1),
                y=data_train["close_future"],
            )

            # Our current situation
            last_row = data.iloc[[-1]]

            X_test = last_row.drop("close_future", axis=1)
            predictions = rf.predict(X_test)

            # Our model's preduicted price
            self.prediction = predictions[0]

            # Calculate the percentage change that the model predicts
            expected_price_change = self.prediction - self.last_price
            self.expected_price_change_pct = expected_price_change / self.last_price

            # Our machine learning model is predicting that the asset will increase in value
            if self.expected_price_change_pct > self.price_change_threshold_up:
                max_position_size = self.max_pct_portfolio * self.portfolio_value
                value_to_trade = self.portfolio_value * self.pct_portfolio_per_trade
                quantity = value_to_trade // self.last_price

                # Check that we are not buying too much of the asset
                if (self.asset_value_potential + value_to_trade) < max_position_size:
                    # Market order
                    main_order = self.create_order(self.symbol, quantity, "buy")
                    self.submit_order(main_order)

            # Our machine learning model is predicting that the asset will decrease in value
            elif self.expected_price_change_pct < -self.price_change_threshold_down:
                max_position_size = self.max_pct_portfolio * self.portfolio_value
                value_to_trade = self.portfolio_value * self.pct_portfolio_per_trade
                quantity = value_to_trade // self.last_price

                # Check that we are not selling too much of the asset
                if (self.asset_value_potential + value_to_trade) < max_position_size:
                    # Market order
                    main_order = self.create_order(self.symbol, quantity, "sell")
                    self.submit_order(main_order)

    def on_abrupt_closing(self):
        self.sell_all()

    def before_market_closes(self):
        self.sell_all()

    def trace_stats(self, context, snapshot_before):
        row = {
            "prediction": self.prediction,
            "last_price": self.last_price,
            "squared_error": (self.prediction - self.last_price) ** 2,
            "expected_price_change_pct": self.expected_price_change_pct,
            "shares_owned_potential": self.shares_owned_potential,
            "asset_value_potential": self.asset_value_potential,
        }

        return row

    # Helper Functions

    def get_data(self, symbol, window_size):
        """Gets pricing data from our data source, then calculates a bunch of technical indicators

        Args:
            symbol (str): The asset symbol that we want the data for
            window_size (int): The amount of data points that we want to get from our data source (in minutes)

        Returns:
            pandas.DataFrame: A DataFrame with the asset's prices and technical indicators
        """
        data_length = window_size + 40

        # bars = self.get_symbol_bars(symbol, data_length, "minute")
        bars = self.get_historical_prices(symbol, data_length, "minute")
        # print("Bars ",bars)
        # print("symbol", symbol)
        # print("data_length", data_length)
        data = bars.df
        # data = bars

        times = data.index.to_series()
        current_datetime = self.get_datetime()
        data["timeofday"] = (times.dt.hour * 60) + times.dt.minute
        data["timeofdaysq"] = ((times.dt.hour * 60) + times.dt.minute) ** 2
        data["unixtime"] = data.index.view(np.int64) // 10 ** 9
        data["unixtimesq"] = data.index.view(np.int64) // 10 ** 8
        data["time_from_now"] = current_datetime.timestamp() - data["unixtime"]
        data["time_from_now_sq"] = data["time_from_now"] ** 2

        data["delta"] = np.append(
            None,
            (np.array(data["close"])[1:] - np.array(data["close"])[:-1])
            / np.array(data["close"])[:-1],
        )
        data["rsi"] = ta.momentum.rsi(data["close"])
        data["ema"] = ta.trend.ema_indicator(data["close"])
        data["cmf"] = ta.volume.chaikin_money_flow(
            data["high"], data["low"], data["close"], data["volume"]
        )
        data["vwap"] = ta.volume.volume_weighted_average_price(
            data["high"], data["low"], data["close"], data["volume"]
        )
        data["bollinger_high"] = ta.volatility.bollinger_hband(data["close"])
        data["bollinger_low"] = ta.volatility.bollinger_lband(data["close"])
        data["macd"] = ta.trend.macd(data["close"])
        # data["adx"] = ta.trend.adx(data["high"], data["low"], data["close"])
        ichimoku = ta.trend.IchimokuIndicator(data["high"], data["low"])
        data["ichimoku_a"] = ichimoku.ichimoku_a()
        data["ichimoku_b"] = ichimoku.ichimoku_b()
        data["ichimoku_base"] = ichimoku.ichimoku_base_line()
        data["ichimoku_conversion"] = ichimoku.ichimoku_conversion_line()
        data["stoch"] = ta.momentum.stoch(data["high"], data["low"], data["close"])
        data["kama"] = ta.momentum.kama(data["close"])
        data = data.dropna()

        data = data.iloc[-window_size:]

        return data


if __name__ == "__main__":
    is_live = False

    if is_live:
        ####
        # Run the strategy
        ####
        ac = AlpacaConfig(True)

        broker = Alpaca(ac)
        strategy = MachineLearning(
            broker=broker,
        )
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()

    else:
        ####
        # Get and Organize Data
        ####
        symbol = "SRNE"
        asset = Asset(symbol=symbol, asset_type="stock")
        df = pd.read_csv(f"data/{asset}_Minute.csv")
        df = df.set_index("time")
        df.index = pd.to_datetime(df.index)

        backtesting_start = datetime(2021, 8, 1)
        backtesting_end = datetime(2021, 8, 10)

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
            pandas_data=my_data,
            symbol=asset,
        )
