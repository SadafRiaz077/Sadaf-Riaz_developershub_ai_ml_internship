Task 2: Predict Future Stock Prices (Short-Term)

DevelopersHub Corporation — AI/ML Engineering Internship


Task Objective

The goal of this task is to use historical stock market data to predict the next day's closing price using supervised machine learning regression models. This involves fetching real financial data via an API, engineering relevant time-series features, training regression models, and comparing their predictive performance.


Dataset Used

Stock Market Data — Apple Inc. (AAPL)


Source: Yahoo Finance, retrieved using the yfinance Python library.
Period: last 2 years of daily data (PERIOD = '2y', INTERVAL = '1d').
Raw columns: Open, High, Low, Close, Volume.
Engineered features added on top of raw data:

Price_Range (High − Low)
Open_Close_Gap (Open − previous day's Close)
Prev_Close, Prev_Volume (lag features)
MA_5, MA_10 (5-day and 10-day moving averages)
Daily_Return (% change day-over-day)



No missing values found in the fetched data.



Models Applied

Two regression models were trained to predict the next day's Close price, using Open, High, Low, Volume, and the engineered features above:


Linear Regression — trained on standardized features (StandardScaler).
Random Forest Regressor (200 trees, max_depth=10, min_samples_split=5) — trained on raw (unscaled) features.


Data was split chronologically (80% train / 20% test, shuffle=False) to respect time-series order and avoid lookahead bias.

Evaluation metrics: MAE, RMSE, R² score, and MAPE.


Key Results and Findings


Random Forest outperformed Linear Regression across all evaluation metrics (MAE, RMSE, R², MAPE), confirming its ability to better capture non-linear relationships and feature interactions present in stock price movement.
Previous day's closing price (Prev_Close) was the single most important feature for both models, which aligns with the well-known fact that stock prices are highly autocorrelated day-to-day (short-term momentum).
Moving averages (MA_5, MA_10) also contributed meaningfully to prediction accuracy, helping smooth out daily noise.
The actual-vs-predicted plots showed that Random Forest predictions tracked the actual closing price curve more tightly than Linear Regression, especially during periods of higher volatility.
Residual plots showed Linear Regression had slightly larger and more patterned errors, while Random Forest residuals were smaller and more randomly scattered — indicating a better model fit.
The daily return distribution was approximately centered around 0%, consistent with normal day-to-day stock price fluctuation, with no extreme skew.
Limitation: Both models predict short-term price movement reasonably well on historical data, but stock prices are inherently noisy and influenced by external/macroeconomic factors not captured in this feature set — so these models are for learning/demonstration purposes, not real trading decisions.



Tools & Libraries

pandas, numpy, yfinance, scikit-learn (LinearRegression, RandomForestRegressor, StandardScaler, train_test_split, metrics), matplotlib, seaborn

Skills Demonstrated

Time-series data handling, feature engineering, regression modeling, API-based data fetching, and visualization of predictions vs actual values.

How to Run

Open Task2_Stock_Price_Prediction.ipynb in Jupyter Notebook or Google Colab and run all cells sequentially. Requires internet access to fetch live data via yfinance. The stock ticker can be changed by editing the TICKER variable (e.g., 'TSLA', 'MSFT', 'GOOGL').
Colab_Notebook link:
https://colab.research.google.com/drive/1993TjTF2iAWGmeyNnepj5mOiiWrKNjvz?usp=sharing

