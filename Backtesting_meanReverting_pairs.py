import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pandas as pd
import seaborn as sns
import yfinance as yf
import datetime as dt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
sns.set_style("darkgrid")


'''
    Creates an objevt of class mean_reverting_pair_trading. 
    Takes the following inputs: 
        df1, df2: Pandas dataframe with same Timedate index column and closing values. 
        Symbols: Tuple of ticker names of the stocks. 
        Lookback : 
'''

class mean_reverting_pair_trading: 
    def __init__(
            self, df1, df2, symbols, lookback_lower = 50, lookback_upper = 150, steps = 25,
            z_entry = 2, z_exit = 1, plot = True): 
        self.df1 = df1
        self.df2 = df2
        self.symbols = symbols 
        self.lookback_lower = lookback_lower
        self.lookback_upper = lookback_upper
        self.steps = steps 
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.plot = plot 
        self.data = self.create_pairs_data(self.df1, self.df2, self.symbols)


    def create_pairs_data(self, df1, df2, symbols):
        print("Constructing dual matrix for %s and %s..." % symbols)
        pairs = pd.DataFrame(index=df1.index)
        pairs['%s_close' % symbols[0].lower()] = df1['close']
        pairs['%s_close' % symbols[1].lower()] = df2['close']
        pairs.index = pd.to_datetime(pairs.index)
        pairs = pairs.dropna()
        return pairs
    
    def calculate_spread_zscore(self, pairs, symbols, lookback=100):
        print("Fitting the rolling Linear Regression...")

        model = RollingOLS(
            endog=pairs[f'{symbols[0].lower()}_close' ],
            exog=sm.add_constant(pairs[f'{symbols[1].lower()}_close']),
            window=lookback
        )
        rres = model.fit()
        params = rres.params.copy()

        # Construct the hedge ratio and eliminate the first

        pairs['hedge_ratio'] = params[f'{symbols[1].lower()}_close']
        pairs.dropna(inplace=True)

        # Create the spread and then a z-score of the spread

        print("Creating the spread/zscore columns...")
        pairs['spread'] = (
            pairs[f'{symbols[0].lower()}_close'] - pairs['hedge_ratio']*pairs[f'{symbols[1].lower()}_close']
        )
        pairs['zscore'] = (
            pairs['spread'] - np.mean(pairs['spread']))/np.std(pairs['spread']
        )
        return pairs
    
    def create_long_short_market_signals(
            self, pairs, symbols, z_entry_threshold=2.0, z_exit_threshold=1.0
        ):

        # Calculate when to be long, short and when to exit
        pairs['longs'] = (pairs['zscore'] <= -z_entry_threshold)*1.0
        pairs['shorts'] = (pairs['zscore'] >= z_entry_threshold)*1.0
        pairs['exits'] = (np.abs(pairs['zscore']) <= z_exit_threshold)*1.0

        pairs['long_market'] = 0.0
        pairs['short_market'] = 0.0

        long_market = 0
        short_market = 0

        print("Calculating when to be in the market (long and short)...")
        for i, b in enumerate(pairs.iterrows()):
            # Calculate longs
            if b[1]['longs'] == 1.0:
                long_market = 1
            # Calculate shorts
            if b[1]['shorts'] == 1.0:
                short_market = 1
            # Calculate exists
            if b[1]['exits'] == 1.0:
                long_market = 0
                short_market = 0
            
            pairs.iloc[i]['long_market'] = long_market
            pairs.iloc[i]['short_market'] = short_market
        return pairs

    def create_portfolio_returns(self, pairs, symbols):
        # Convenience variables for symbols
        sym1 = symbols[0].lower()
        sym2 = symbols[1].lower()

        # Construct the portfolio object with positions information
        # Note the minuses to keep track of shorts!
        print("Constructing a portfolio...")
        portfolio = pd.DataFrame(index=pairs.index)
        portfolio['positions'] = pairs['long_market'] - pairs['short_market']
        portfolio[sym1] = -1.0 * pairs['%s_close' % sym1] * portfolio['positions']
        portfolio[sym2] = pairs['%s_close' % sym2] * portfolio['positions']
        portfolio['total'] = portfolio[sym1] + portfolio[sym2]

        # Construct a percentage returns stream and eliminate all
        # of the NaN and -inf/+inf cells
        print("Constructing the equity curve...")
        portfolio['returns'] = portfolio['total'].pct_change()
        portfolio['returns'].fillna(0.0, inplace=True)
        portfolio['returns'].replace([np.inf, -np.inf], 0.0, inplace=True)
        portfolio['returns'].replace(-1.0, 0.0, inplace=True)

        # Calculate the full equity curve
        portfolio['returns'] = (portfolio['returns'] + 1.0).cumprod()
        return portfolio

    def run_backtest(self, lookback_lower = 50, lookback_upper = 150, steps = 25):
        lookbacks = range(self.lookback_lower, self.lookback_upper, self.steps)
        returns = []
        for lb in lookbacks:
            print("Calculating lookback=%s..." % lb)
            pairs = self.create_pairs_data(SPY, DIA, ('SPY', 'DIA'))
            pairs = self.calculate_spread_zscore(pairs, self.symbols, lookback=lb)
            pairs = self.create_long_short_market_signals(
                pairs = pairs, symbols = self.symbols, z_entry_threshold= self.z_entry, z_exit_threshold=1.0
            )
            portfolio = self.create_portfolio_returns(pairs, self.symbols)
            returns.append(portfolio.iloc[-1]['returns'])

        print("Plot the lookback-performance scatterchart...")
        plt.plot(lookbacks, returns, '-o')
        plt.show()
        if self.plot: 
            fig = plt.figure()

            ax1 = fig.add_subplot(211,  ylabel='%s growth (%%)' % self.symbols[0])
            (pairs['%s_close' % self.symbols[0].lower()].pct_change()+1.0).cumprod().plot(ax=ax1, color='r', lw=2.)

            ax2 = fig.add_subplot(212, ylabel='Portfolio value growth (%%)')
            portfolio['returns'].plot(ax=ax2, lw=2.)

            plt.show()