Got it — here's the clean version without emojis:
Task 1: Exploring and Visualizing the Iris Dataset
DevelopersHub Corporation — AI/ML Engineering Internship
Task Objective
The goal of this task is to load, inspect, and visualize the Iris dataset in order to understand data trends, feature distributions, and relationships between variables across the three flower species. This task builds foundational skills in data loading, exploratory data analysis (EDA), and visualization using pandas, matplotlib, and seaborn.
Dataset Used
Iris Dataset

Loaded directly via seaborn.load_dataset('iris') — no manual download required
150 rows × 5 columns
Features: sepal_length, sepal_width, petal_length, petal_width (numeric, cm)
Target: species (setosa, versicolor, virginica) — 50 samples each
No missing values — clean, balanced dataset

Models Applied
This task is exploratory/visualization-based, not predictive modeling. Techniques applied:

Descriptive statistics (.describe(), .info(), .groupby().mean())
Bar chart — species count
Histograms — overall and per-species distributions
Scatter plots — sepal length vs petal length; petal length vs petal width
Box plots — outlier detection
Violin plots — distribution shape per species
Correlation heatmap
Pair plot — all feature combinations

Key Results and Findings

Perfectly balanced dataset — 50 samples/species, zero missing values
Setosa is clearly separable from the other species, especially via petal measurements
Versicolor and Virginica overlap — harder to distinguish, would need a classifier for clean separation
Petal length and petal width are the most correlated and most discriminative features
Sepal width is the weakest/least useful feature for separating species
Minor outliers found in sepal width; no major outlier issues elsewhere
Conclusion: petal-based features are stronger predictors of species than sepal-based features

Tools & Libraries
pandas, numpy, matplotlib, seaborn
Skills Demonstrated
Data loading and inspection, descriptive statistics, EDA, data visualization
How to Run
Open Task1_Iris_Clean.ipynb in Jupyter Notebook or Google Colab and run all cells sequentially. No API keys needed.

Task 2: Predict Future Stock Prices (Short-Term)
DevelopersHub Corporation — AI/ML Engineering Internship
Task Objective
Use historical stock market data to predict the next day's closing price using supervised regression models — covering API-based data fetching, time-series feature engineering, model training, and performance comparison.
Dataset Used
Stock Market Data — Apple Inc. (AAPL)

Source: Yahoo Finance, via the yfinance Python library
Period: last 2 years, daily interval
Raw columns: Open, High, Low, Close, Volume
Engineered features:

Price_Range (High − Low)
Open_Close_Gap (Open − previous Close)
Prev_Close, Prev_Volume (lag features)
MA_5, MA_10 (moving averages)
Daily_Return (% change)


No missing values found

Models Applied

Linear Regression — trained on standardized features (StandardScaler)
Random Forest Regressor (200 trees, max_depth=10, min_samples_split=5)

Data split chronologically (80/20, shuffle=False) to preserve time order. Evaluated using MAE, RMSE, R², MAPE.
Key Results and Findings

Random Forest outperformed Linear Regression across all metrics — better at capturing non-linear patterns
Previous day's Close (Prev_Close) was the single most important feature for both models — confirms strong day-to-day autocorrelation in stock prices
Moving averages (MA_5, MA_10) added meaningful predictive power by smoothing daily noise
Random Forest tracked actual price curves more tightly, especially during volatile periods
Residuals for Random Forest were smaller and more randomly scattered vs. Linear Regression's larger, patterned errors
Daily returns were roughly centered around 0%, consistent with normal market fluctuation
Limitation: Models work well on historical patterns but don't account for external/macro factors — built for learning purposes, not real trading

Tools & Libraries
pandas, numpy, yfinance, scikit-learn, matplotlib, seaborn
Skills Demonstrated
Time-series handling, feature engineering, regression modeling, API data fetching, prediction visualization
How to Run
Open Task2_Stock_Price_Prediction.ipynb in Jupyter/Colab and run all cells. Requires internet access for yfinance. Change the TICKER variable to try other stocks (e.g. TSLA, MSFT).
Colab_Notebook link:
https://colab.research.google.com/drive/1993TjTF2iAWGmeyNnepj5mOiiWrKNjvz?usp=sharing

