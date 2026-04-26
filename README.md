# 🏪 Retail Store Revenue Prediction & Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.2+-green.svg)

## 📌 Project Overview
This project demonstrates an end-to-end Machine Learning pipeline aimed at predicting retail store revenue based on physical and operational characteristics. Rather than simply applying algorithms, this project focuses heavily on **Exploratory Data Analysis (EDA), Feature Importance, and Model Diagnostics** to extract actionable business intelligence.

The goal was to answer a critical business question: *What physical properties of a store actually drive revenue?*

## 🚀 The Machine Learning Workflow

### 1. Data Cleaning & Preprocessing
Real-world data is messy. I implemented a robust preprocessing pipeline to handle anomalies:
* **String Parsing:** Stripped commas and whitespace from financial data to convert string-based `Revenue` into numeric types.
* **Imputation:** Handled missing `Checkout Number` values using median imputation grouped by store `Type`.
* **Target Transformation:** Applied a `log1p` (Logarithmic base e + 1) transformation to the target variable (`Revenue`) to normalize the highly right-skewed financial data.

<img width="3000" height="1800" alt="revenue_distribution" src="https://github.com/user-attachments/assets/d592f5ef-e0cb-4d6e-99c6-65305f9cc352" />

### 2. Feature Engineering
Machine learning models require mathematical representations of categories. I engineered features using:
* **One-Hot Encoding:** Converted nominal categories (`Property`, `Type`) into binary vectors.
* **Binary Mapping:** Mapped ordinal categories (`Old/New`) to 0/1 integers.

### 3. Feature Importance vs. Linear Correlation
I ran both a Correlation Matrix and a Random Forest Regressor to compare linear relationships vs. non-linear interactions.

<img width="3600" height="2400" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/f02b5429-cd37-4d00-970a-e640efd60d7d" />


**The Business Insight:**
While the correlation matrix suggested `Checkout Number` was the strongest predictor (0.77), the Tree-based Feature Importance algorithm revealed the deeper truth:
* **`AreaStore` (Square Footage) drives 61.5% of the model's decision-making.**
* **`Checkout Number` drives 23.8%.**
* **Property Ownership (`Property_Owned`) drives 6.2%.**

*The algorithm proved that store "Type" (Hyper vs Express) is largely redundant if the square footage is already known.*

<img width="3600" height="2100" alt="feature_importance" src="https://github.com/user-attachments/assets/15e6f649-6e06-415b-8cd1-4bd3e38cf532" />

### 4. Model Tuning & Evaluation
I established a baseline using a default `RandomForestRegressor` and then optimized it using `GridSearchCV`. 

**Grid Search Parameters Tested:**
* `n_estimators`: [50, 100, 200]
* `max_depth`: [None, 10, 20]
* `min_samples_split`: [2, 5, 10]

**Optimal Parameters Found:** `{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 200}`

## 📊 Results & "The Data Wall"
* **Optimized R² Score:** `0.4207`
* **Real-World MAE:** `$8,973,871.09`

**Model Diagnostic Conclusion:** 
The model explains ~42% of the variance in revenue. While the model successfully identified the primary physical drivers of revenue (Area and Checkouts), it hit a "Data Wall." The dataset consists of only 118 records. 

**Next Steps for Production:** 
To cross the 0.80 R² threshold and reduce the MAE to acceptable production limits, the model requires a larger sample size (>1,000 stores) and external feature injection, specifically:
1. **Geospatial Data** (Location, City Population)
2. **Foot Traffic Metrics**
3. **Marketing Spend / Promotions**

## 💻 How to Run
1. Clone the repository.
2. Ensure you have `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` installed.
3. Run the Jupyter Notebook to step through the data cleaning, feature importance plotting, and grid search pipeline.

---
*This project was built to showcase rigorous ML diagnostic procedures, hyperparameter tuning, and business-focused data storytelling.*
