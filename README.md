# Diabetes Dataset — Data Preprocessing

A hands-on data preprocessing exercise using the Pima Indians Diabetes dataset, covering the full pipeline from raw data inspection to feature scaling — preparing data for machine learning models.

## 📌 Objective

Practice the end-to-end preprocessing workflow:
- Inspect and understand raw data
- Detect and remove outliers
- Analyze feature correlations
- Check class balance of target variable
- Encode and scale features for ML readiness

## 🗂 Dataset

**Pima Indians Diabetes Dataset** — 768 rows, 9 columns including features like Glucose, BloodPressure, BMI, Insulin, Age and a binary target variable `Outcome` (1 = Diabetes, 0 = No Diabetes).

> Note: This dataset contains hidden missing values disguised as 0s in columns like Glucose, BloodPressure and BMI — which are medically impossible as zero and were treated accordingly.

## 🛠 Steps Performed

1. **Load & inspect data** — `df.info()`, `df.head()`
2. **Statistical summary & outlier visualization** — `df.describe()`, box plots for every column using matplotlib subplots
3. **Outlier removal using IQR method** — calculated Q1, Q3 and IQR for the Insulin column; removed values beyond `Q1 - 1.5×IQR` and `Q3 + 1.5×IQR`
4. **Correlation analysis** — computed pairwise correlation matrix using `df.corr()`, visualized as a heatmap using seaborn; sorted features by correlation with target variable
5. **Target variable distribution** — pie chart to check class balance between Diabetes vs No Diabetes
6. **Feature scaling:**
   - **Normalization** (MinMaxScaler) — rescaled features to range [0, 1]
   - **Standardization** (StandardScaler) — transformed features to mean=0 and std=1

## 📊 Key Findings

- Insulin column had the most extreme outliers — IQR method used as data is right-skewed
- Glucose was the most strongly correlated feature with Outcome (~0.47)
- BloodPressure showed very weak correlation with Outcome (~0.07)
- Target variable showed approximately 65% No Diabetes vs 35% Diabetes — slightly imbalanced
- Hidden zeros found in columns like Glucose, BloodPressure and BMI — medically impossible values treated as missing data

## 🧰 Tools & Libraries Used

- Python (Google Colab)
- pandas — data loading and manipulation
- numpy — numerical operations and percentile calculations
- matplotlib — box plots and pie charts
- seaborn — correlation heatmap
- scikit-learn — MinMaxScaler, StandardScaler

## 📁 Files

- `diabetes_preprocessing.ipynb` — full notebook with code, outputs and comments

## 🎯 What I Learned

- IQR method is better than Z-score (mean ± 2std) for skewed data because Q1/Q3 are not affected by extreme values
- Difference between normalization (scales to 0–1 range) and standardization (scales to mean=0, std=1) and when to use each
- Correlation analysis helps identify which features are most useful for predicting the target variable
- Class imbalance in the target variable can bias ML models toward the majority class
- Hidden missing values (zeros) are a unique challenge in medical datasets — not all missing data shows up as NaN

## 🔗 Reference

Exercise based on [GeeksforGeeks — Data Preprocessing](https://www.geeksforgeeks.org/data-analysis/)
