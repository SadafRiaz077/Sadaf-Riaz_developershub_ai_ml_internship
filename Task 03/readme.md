Task 3: Heart Disease Prediction

DevelopersHub Corporation — AI/ML Engineering Internship


Task Objective

The goal of this task is to build a machine learning classification model that predicts whether a patient is at risk of heart disease based on their clinical and health data. This includes data cleaning, exploratory data analysis, model training, and evaluation using accuracy, ROC-AUC, and confusion matrix, along with identifying the most important features driving the prediction.


Dataset Used

Heart Disease UCI Dataset


Loaded directly from a public GitHub source (sharmaroshan/Heart-UCI-Dataset), eliminating the need for manual download.
Contains patient health records with features such as: age, sex, cp (chest pain type), trestbps (resting blood pressure), chol (cholesterol), fbs (fasting blood sugar), restecg, thalach (max heart rate), exang (exercise-induced angina), oldpeak (ST depression), slope, ca (number of major vessels), thal.
Target: target — converted to binary (0 = No Heart Disease, 1 = Heart Disease).
Data cleaning steps: checked for missing values (none found), checked and removed duplicate rows, and confirmed the target variable was properly binarized.



Models Applied

Two classification models were trained and compared:


Logistic Regression (max_iter=1000, solver='lbfgs') — trained on standardized features (StandardScaler).
Decision Tree Classifier (max_depth=5, min_samples_split=10, min_samples_leaf=5, criterion = Gini) — trained on raw (unscaled) features.


Data was split using an 80/20 stratified train-test split to preserve class balance, and validated further using 5-fold stratified cross-validation.

Evaluation metrics: Accuracy, ROC-AUC score, Confusion Matrix, Classification Report (precision/recall/F1), and cross-validation accuracy (mean ± std).


Key Results and Findings


The dataset was clean (no missing values) and reasonably balanced between disease/no-disease classes, making it well-suited for direct classification modeling without heavy preprocessing.
Both Logistic Regression and Decision Tree achieved strong, comparable accuracy and ROC-AUC scores on the test set; cross-validation confirmed these results were stable rather than a result of a lucky train/test split (low standard deviation across folds).
Males showed a significantly higher rate of heart disease than females in this dataset, consistent with known epidemiological trends.
Asymptomatic chest pain (cp = 3) was the chest pain category most strongly associated with a positive heart disease diagnosis — somewhat counterintuitive, but a well-documented pattern in cardiology, since "silent" symptoms can indicate more advanced disease.
Feature importance analysis (Decision Tree) and coefficient analysis (Logistic Regression) both highlighted chest pain type (cp), maximum heart rate (thalach), ST depression (oldpeak), and number of major vessels (ca) as the top predictive features — all clinically meaningful indicators of cardiac health.
The ROC curves for both models showed strong separation from the random-guess baseline, with AUC scores well above 0.5, indicating good discriminative ability.
The confusion matrices showed a relatively low rate of both false positives and false negatives for both models, though Logistic Regression had a slight edge in overall balance between precision and recall.
Conclusion: Logistic Regression is a strong, interpretable baseline for this problem, while the Decision Tree offers an easily visualized, rule-based alternative — both reaching clinically reasonable predictive performance on this dataset.



Tools & Libraries

pandas, numpy, scikit-learn (LogisticRegression, DecisionTreeClassifier, StandardScaler, train_test_split, StratifiedKFold, cross_val_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report), matplotlib, seaborn

Skills Demonstrated

Binary classification, medical/clinical data interpretation, model evaluation using ROC-AUC and confusion matrix, cross-validation, and feature importance analysis.

How to Run

Open Task3_Heart_Disease_Prediction.ipynb in Jupyter Notebook or Google Colab and run all cells sequentially. Requires internet access to load the dataset from GitHub; no API keys are needed.
Colab_Notebook link:
https://colab.research.google.com/drive/1oBrURu93iuMc9ePSv5yPwKW5KJ-miKPn?usp=sharing
