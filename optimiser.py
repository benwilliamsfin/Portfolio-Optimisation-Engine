
# Portfolio Optimization Engine
# Author: Ben Williams
# Description: A vectorised Markowitz Portfolio Optimization engine built to 
#              demonstrate quantitative finance and performance engineering principles.

# Library Imports and Dependancy Setups

import numpy as np  # Library for maxtrix vectorisation and linear algebra
import pandas as pd # Data structure library for financial time-series data
import yfinance as yf # Ingest histrorical market price data

# Step 1: Define a mutli-asset universe to capture broad macroeconomic exposures
tickers = ['SPY', 'TLT', 'GLD', 'QQQ']
# SPY - US Large-Cap Equities - Core Equity Beta
# TLT - Long-term US Treasury Bonds - Interest rate sensitivity + Fixed Income Hedge
# GLD - Physical Commodity - Inflation Hedge + Alternative source of value
# QQQ - Nasdaq-100 Tech Growth - High-Beta growth factor

start_date = "2021-01-01"
end_date = "2026-01-01"
# Select a 5-year histrorical timeframe to capture macroeconomic trends, in particular post-covid recovery period
# rate hiking cycles, and subsequent inflation/tech market cycles

print("Fetching Historical Asset Data")
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
# auto_adjust=False ensures we pull the legacy multi-index adj close structure that accurately reflects
# historical dividend and corporate action adjustments

print(data.head())
# Print the first few rows to verify the table

# Step 2: Return Calculations & Annualisation
# Convert the raw price series data into daily % returns: 
#                                                       R_t = (P_t - P_{t-1}) / P_{t-1}

raw_returns = data.pct_change()
print(raw_returns.head())
# Note the first row of NaN values, we must eliminate these

returns = raw_returns.dropna()
print(returns.head())                                                                                                                                                      
# We have eliminiated the first row of NaN values

# Annualise expected returns by multiplying daily mean returns by 252 trading days 
    #(Standard convention for annualising trading data in financial markets)
mean_returns = returns.mean() * 252

# Compute annualised covariance matrix (sigma) across asset universe by scaling
# daily covariance by 252. This matrix models both individual asset variances and
# cross-asset co-movement required for Markowitz risk modelling.
cov_matrix = returns.cov() * 252

# Verification
print("\nData Ingestion and Covariance Verification")
print("\nAnnualised Mean Returns")
print(mean_returns)
print ("\nCovariance Matrix (Assets x Assets)", cov_matrix.shape)
#\n drops a line prior to printing text, purely visual clarity
