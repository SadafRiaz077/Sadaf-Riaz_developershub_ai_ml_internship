Task 1: Exploring and Visualizing the Iris Dataset

DevelopersHub Corporation — AI/ML Engineering Internship


Task Objective

The goal of this task is to load, inspect, and visualize the Iris dataset in order to understand data trends, feature distributions, and relationships between variables across the three flower species. This task builds foundational skills in data loading, exploratory data analysis (EDA), and visualization using pandas, matplotlib, and seaborn.


Dataset Used

Iris Dataset


Loaded directly via seaborn.load_dataset('iris') (no manual download required).
150 rows × 5 columns.
Features: sepal_length, sepal_width, petal_length, petal_width (all numeric, in cm).
Target: species (categorical — setosa, versicolor, virginica), 50 samples each.
No missing values; dataset is clean and balanced.



Models Applied

This task is exploratory/visualization-based and does not involve training a predictive model. Instead, the following analytical and visualization techniques were applied:


Descriptive statistics (.describe(), .info(), .groupby().mean())
Visual exploration techniques:

Bar chart (species count)
Histograms (overall and per-species feature distributions)
Scatter plots (sepal length vs petal length; petal length vs petal width)
Box plots (outlier detection per feature/species)
Violin plots (distribution shape per species)
Correlation heatmap
Pair plot (all feature combinations)
Mean feature comparison bar chart






Key Results and Findings


The dataset is perfectly balanced, with exactly 50 samples per species and zero missing values, making it ready for analysis/modeling without cleaning.
Iris Setosa is clearly and easily separable from the other two species, especially using petal length and petal width — it forms a distinct, non-overlapping cluster in scatter plots.
Versicolor and Virginica show partial overlap in feature space (particularly in sepal measurements), meaning they are harder to distinguish than Setosa and would require more refined features or a classifier for separation.
Petal length and petal width are the most strongly correlated features with each other and are also the most discriminative between species — they show the clearest separation in box/violin plots and the highest correlation coefficient in the heatmap.
Sepal width has the weakest correlation with the other features and shows more overlapping distributions across species, making it the least useful feature for distinguishing species visually.
Box plots revealed a small number of outliers in sepal width, but no major outlier issues elsewhere.
Overall, petal-based features are better predictors of species than sepal-based features — a useful insight if this dataset were later used for classification.



Tools & Libraries

pandas, numpy, matplotlib, seaborn

Skills Demonstrated

Data loading and inspection, descriptive statistics, exploratory data analysis (EDA), and data visualization.

How to Run

Open Task1_Iris_Clean.ipynb in Jupyter Notebook or Google Colab and run all cells sequentially. No API keys or external downloads are required — the dataset loads directly from seaborn.
Colab Notebook Link:
https://colab.research.google.com/drive/1FLed5oMAYXLZ9V_7seA6tFmIMCb-EHCh?usp=sharing
