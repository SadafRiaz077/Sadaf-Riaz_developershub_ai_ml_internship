
Task 6: House Price Prediction

DevelopersHub Corporation — AI/ML Engineering Internship


Task Objective

Predict house sale prices using property features such as size, number of bedrooms, quality ratings, and location. Two regression models — Linear Regression and Gradient Boosting — are trained and compared using standard regression evaluation metrics (MAE, RMSE, and R²). The project covers the full ML pipeline: data loading, EDA, preprocessing, model training, evaluation, visualisation, and feature importance analysis.


Dataset Used

House Prices – Advanced Regression Techniques (Kaggle)

PropertyDetailsSourceKaggle CompetitionFiletrain.csvRows1,460 housesColumns81 (80 features + 1 target)Target variableSalePrice (house sale price in USD)Feature typesNumeric (area, counts, year) + Categorical (neighbourhood, quality ratings, building type)

The dataset covers residential properties in Ames, Iowa, USA, and includes features ranging from square footage and garage capacity to basement quality and neighbourhood classification.


Models Applied

ModelLibraryKey HyperparametersLinear Regressionsklearn.linear_model.LinearRegressionDefaultGradient Boosting Regressorsklearn.ensemble.GradientBoostingRegressorn_estimators=100, learning_rate=0.1, max_depth=3, random_state=42

Both models were trained on the same preprocessed feature set and evaluated on the same held-out test split (80/20 train-test split).


Approach

Step-by-Step Pipeline

StepAction1Installed and imported all libraries (pandas, numpy, matplotlib, seaborn, scikit-learn)2Loaded Kaggle train.csv (1,460 rows, 81 columns)3EDA — price distribution, missing value analysis, scatter plots, box plots, histograms, correlation heatmap4Preprocessing — missing value imputation, label encoding of categoricals, feature engineering (e.g. TotalSF), StandardScaler5Trained Linear Regression + Gradient Boosting6Evaluated both models with MAE, RMSE, and R²7Visualised actual vs predicted prices (scatter plots, line plots, residual plots, metrics bar chart)8Feature importance analysis (top 20 features from Gradient Boosting)95-fold Cross-Validation for both models10Final results summary and key insights

Preprocessing Details


Missing values: Numeric columns filled with median; categorical columns filled with mode or "None" for features where absence is meaningful (e.g. PoolQC, Alley).
Categorical encoding: LabelEncoder applied to all object-type columns.
Feature engineering: TotalSF (total square footage = basement + 1st floor + 2nd floor) created as an additional feature.
Feature scaling: StandardScaler applied to all features before model training.
Target transformation: SalePrice analysed in both raw and log-transformed form during EDA to check for skewness.



Key Results and Findings

Model Performance

ModelMAE ($)RMSE ($)R² ScoreLinear Regression~22,000~35,000~0.82Gradient Boosting~15,000~24,000~0.91


Note: Exact values depend on the train-test random split. The table above shows approximate expected ranges.



** Best Model: Gradient Boosting Regressor** — significantly outperforms Linear Regression on all three metrics by capturing non-linear relationships in the data.

Cross-Validation Results (5-Fold R²)

ModelMean R²StdLinear Regression~0.80±0.03Gradient Boosting~0.88±0.02

Top 5 Most Important Features (Gradient Boosting)


OverallQual — Overall material and finish quality (strongest single predictor)
GrLivArea — Above-ground living area (square feet)
TotalSF — Total square footage (engineered feature)
Neighborhood — Physical location / neighbourhood in Ames
YearBuilt / HouseAge — Age of the property


Key Insights


Overall quality rating (OverallQual) is the most powerful predictor of sale price — buyers pay a premium for well-finished homes.
Size matters most among numeric features. GrLivArea and TotalSF are the top size-based predictors.
Gradient Boosting significantly outperforms Linear Regression by capturing non-linear patterns (e.g. exponential price growth for high-quality properties).
Location drives price — Neighborhood is consistently among the top predictors.
Newer homes command higher prices — YearBuilt / HouseAge is a strong signal.
Linear Regression struggles with luxury homes — its residuals are larger at the high end of the price range, while Gradient Boosting handles outliers better.


Skills Demonstrated


Regression modeling — Linear Regression and Gradient Boosting trained and compared on a real estate dataset
Feature scaling and selection — StandardScaler, LabelEncoder, and engineered TotalSF feature
Model evaluation (MAE, RMSE, R²) — All three standard regression metrics computed and visualised with a comparative bar chart
Real estate data understanding — Correct interpretation of domain-specific features (OverallQual, GrLivArea, Neighborhood, YearBuilt)



Visualisations Produced

FileDescriptionsaleprice_distribution.pngRaw and log-transformed SalePrice distributionsmissing_values.pngBar chart of missing value percentages per columnscatter_plots.pngKey feature vs SalePrice scatter plotsbox_plots.pngBox plots of top categorical features vs SalePricecorrelation_heatmap.pngHeatmap of feature correlationsactual_vs_predicted.pngScatter and line plots: actual vs predicted prices (both models)residual_analysis.pngResidual vs predicted and residual distribution plotsmetrics_comparison.pngSide-by-side MAE, RMSE, and R² bar chartfeature_importance.pngTop 20 features by Gradient Boosting importance scorecross_validation.png5-fold CV R² scores per fold for both models


How to Run

Requirements

bashpip install pandas numpy matplotlib seaborn scikit-learn

Environment


Google Colab (CPU runtime is sufficient — no GPU needed for this task)
Download train.csv from Kaggle before running


Steps


Open Task6_House_Price_Prediction.ipynb in Google Colab.
Run Step 2 — when prompted, upload your train.csv file from Kaggle.
Run all remaining cells sequentially (Steps 3–10).



Project Structure

Task6_House_Price_Prediction.ipynb ← Main notebook
train.csv ← Dataset (download from Kaggle, not included in repo)
saleprice_distribution.png ← EDA visualisation
missing_values.png ← EDA visualisation
actual_vs_predicted.png ← Model output visualisation
residual_analysis.png ← Model evaluation visualisation
metrics_comparison.png ← Model comparison chart
feature_importance.png ← Feature importance chart
cross_validation.png ← Cross-validation results


References


Kaggle House Prices Competition
Scikit-Learn GradientBoostingRegressor
Scikit-Learn LinearRegression
Colab Notebook link:
https://colab.research.google.com/drive/1kG3ZUm3thdfdxn0M7evRhA6i4iCPm7Cg?usp=sharing
