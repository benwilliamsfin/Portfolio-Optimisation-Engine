
# Portfolio Optimization Engine
# Author: Ben Williams
# Description: A vectorised Markowitz Portfolio Optimization engine built to 
#              demonstrate quantitative finance and performance engineering principles.

# Library Imports and Dependancy Setups

import numpy as np  # Library for maxtrix vectorisation and linear algebra
import pandas as pd # Data structure library for financial time-series data
import yfinance as yf # Ingest histrorical market price data

# Step 1: Define a mutli-asset universe to capture broad macroeconomic exposures
# tickers = ['SPY', 'TLT', 'GLD', 'QQQ'] = a list of strings identifying the assets we want to include in our portfolio optimization engine.
tickers = ['SPY', 'TLT', 'GLD', 'QQQ']
# SPY - US Large-Cap Equities - Core Equity Beta
# TLT - Long-term US Treasury Bonds - Interest rate sensitivity + Fixed Income Hedge
# GLD - Physical Commodity - Inflation Hedge + Alternative source of value
# QQQ - Nasdaq-100 Tech Growth - High-Beta growth factor

start_date = "2021-01-01"
end_date = "2026-01-01"
# 5-Year histrorical timeframe to capture macroeconomic trends, in particular post-covid recovery period
# rate hiking cycles, and subsequent inflation/tech market cycles
# This 5-year window serves as a macro stress-test regime. Historical mean returns are used as an empirical baseline proxy, 
# acknowledging the classic out-of-sample estimation noise inherent in backward-looking expected returns.

print("Fetching Historical Asset Data")
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
# auto_adjust=False ensures we pull the legacy multi-index adj close structure that accurately reflects
# historical dividend and corporate action adjustments

print(data.head())
# Print the first few rows to verify the table 
# .head() method returns the first 5 rows of the dataframe by default.

# Step 2: Return Calculations & Annualisation
# Convert the raw price series data into daily % returns: 
#                                                       R_t = (P_t - P_{t-1}) / P_{t-1}

raw_returns = data.pct_change()
print(raw_returns.head())
# Note the first row of NaN values, we must eliminate these

returns = raw_returns.dropna()
print(returns.head())                                                                                                                                                      
# We have eliminiated the first row of NaN values, as t=0 lacks a prior price (t=-1) to compute a return. 
# This is a standard data cleaning step in financial time-series analysis.

mean_returns = returns.mean() * 252
# Annualise expected returns by multiplying daily mean returns by 252 trading days 
# (Standard convention for annualising trading data in financial markets)

# Compute annualised covariance matrix (sigma) across asset universe by scaling
# daily covariance by 252. This matrix models both individual asset variances and
# cross-asset co-movement required for Markowitz risk modelling.
cov_matrix = returns.cov() * 252
# Standard Markowitz assumes covariance stationarity over the period. 
# In live markets, cross-asset correlations are dynamic and tend toward 1.0 during liquidity shocks.
# .cov() method computes the covariance matrix of the returns dataframe, 
# which captures the variance and covariance between each pair of assets in our portfolio.

# Verification
print("\nData Ingestion and Covariance Verification")
print("\nAnnualised Mean Returns")
print(mean_returns)
print ("\nCovariance Matrix (Assets x Assets)", cov_matrix.shape)
# \n drops a line prior to printing text, purely visual clarity
# 4 assets in our universe, so covariance matrix is 4x4.


# Step 3 - Implementation 
# Simulating portfolios using a standard python for-loop. 

import time 

num_portfolios_loop = 10_000
# underscore is a visual separator for large numbers, purely aesthetic.
loop_returns = []
loop_volatilities = []
loop_sharpes = []

print(f"\n[BENCHMARK] Executing simulation across {num_portfolios_loop:,} portfolios")
start_time = time.time()

for i in range (num_portfolios_loop):
    # Generate random weights
    w = np.random.random(len(tickers))
    # Generate a random weight for each asset in the portfolio, creating a vector of length equal to the number of tickers.
    w /= np.sum(w)
    # np.sum(w) computes the sum of the weights, and we divide each weight by this sum to normalise the weights so that they sum to 1.

    # Compute return and volatility
    ret = np.dot(w, mean_returns)
    # np.dot(w, mean_returns) computes the dot product of the weight vector and the mean returns vector, 
    # yielding the expected return of the portfolio.
    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    # np.dot(w.T, np.dot(cov_matrix, w)) computes the portfolio variance using the covariance matrix and the weight vector.
    sharpe = ret/vol 
    # The Sharpe ratio is computed as the expected return divided by the portfolio volatility,
    # providing a measure of risk-adjusted return.

    loop_returns.append(ret)
    # Append the computed return to the loop_returns list
    loop_volatilities.append(vol)
    loop_sharpes.append(sharpe)

loop_duration = time.time() - start_time
print(f"Iterative Loop Finished in: {loop_duration:.4f} seconds for {num_portfolios_loop:,} portfolios.")
# When ran, this takes ~ 1.7 seconds for 10,000 portfolios on my cpu (i7 6700). 
# This is a reasonable benchmark for a single-threaded python implementation, but the iterative overhead scales poorly. 
# Scaling to 1,000,000+ portfolios or adding complex constraints makes this unviable for production systems.
# (1.7/10,000) * 1,000,000 = 170 seconds (roughly 3 minutes) for 1,000,000 portfolios. 
# This is a linear scaling issue that is not acceptable for production systems.
# We must craft a better solution that leverages vectorisation and matrix algebra to eliminate the iterative overhead.
print("DESIGN FLAW IDENTIFIED: Iterative overhead scales poorly. Scaling to 1,000,000+ portfolios or " \
      "adding complex constraints makes this unviable for production systems.")

# Step 4 - Vectorised Portfolio Simulation
# To achieve a more efficient solution, we will leverage numpy's vectorisation capabilities to eliminate the iterative overhead.

num_portfolios = 1_000_000 
#Scaling up 100x larger to demonstrate the performance benefits of vectorisation.
np.random.seed(90) 
# Set a random seed for reproducibility
# Arrived at 90 using random number generator between 1 and 100, but any integer will work.

print(f"\n[OPTIMISED] Executing simulation across {num_portfolios:,} portfolios")
vector_start_time = time.time()

# Generate random weights for all portfolios at once
w_matrix = np.random.random((num_portfolios, len(tickers)))
# Instead of generating weights one portfolio at a time, we generate a 2D array (matrix) of random weights for all portfolios at once.
w_matrix /= np.sum(w_matrix, axis=1, keepdims=True)
# Axis = 1 means we sum across rows (i.e., for each portfolio), and keepdims=True ensures the result is a 2D array, 
# allowing for proper broadcasting during division. 
# (Different array shapes (1D vs 2D) can lead to broadcasting errors if not handled correctly.)

# Vectorised return and volatility calculations
portfolio_returns = np.dot(w_matrix, mean_returns)
# Compute portfolio returns for all portfolios at once using matrix multiplication

portfolio_volatilities = np.sqrt(np.sum(np.dot(w_matrix, cov_matrix) * w_matrix, axis=1))
# Compute portfolio volatilities for all portfolios at once using matrix multiplication

risk_free_rate = 0.0
# Assuming a baseline risk-free rate of 0% for simplicity. 
# In practice, this would be set to the yield of a risk-free asset (e.g., US Treasury Bills).

portfolio_sharpe = (portfolio_returns - risk_free_rate) / portfolio_volatilities
# Excess return per unit of risk (Sharpe Ratio) for all portfolios at once

vector_duration = time.time() - vector_start_time
print(f"Vectorised Simulation Finished in: {vector_duration:.4f} seconds for {num_portfolios:,} portfolios.")

# Vectorised Simulation finished in: 0.1181 seconds for 1,000,000 portfolios.
# This is a substantial improvement over the iterative loop, demonstrating the power of vectorisation and matrix algebra in 
# quantitative finance applications.

# Comparison:
# Iterative Loop: 1.7 seconds for 10,000 portfolios
    # = 0.00017 seconds per portfolio
# Vectorised Simulation: 0.1181 seconds for 1,000,000 portfolios
    # = 0.0000001181 seconds per portfolio