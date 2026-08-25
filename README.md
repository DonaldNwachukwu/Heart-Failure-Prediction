# Heart Failure Prediction — Feature Engineering & Data Preprocessing


##  📌 Project Overview

This project prepares the Heart Failure Prediction Dataset for machine learning by applying a full preprocessing pipeline — data cleaning, feature engineering, encoding, scaling, outlier treatment, and feature selection. 
  
## Business Problem

Heart failure occurs when the heart cannot pump enough blood to meet the body's needs. It’s usually triggered by other health conditions affecting the heart muscles, including coronary artery disease, heart attack, high blood pressure, heart valve disorders, congenital defects, cardiomyopathy, myocarditis, diabetes, and pulmonary hypertension.

While heart failure is often caused by underlying health conditions, each individual’s health risks are unique. Many factors like age, family history, and lifestyle can contribute, but early detection with advanced diagnostic tools can help manage and reduce the likelihood of serious damage. 

Therefore, a heart failure prediction model would be effective in detecting the likelihood or the presence of heart failure in individuals. 


## Objective

The objective is to predict the likelihood of a patient developing heart disease based on 11 clinical and diagnostic features, producing a dataset that is clean, well-structured, and ready for predictive modelling.

## 📂 Repository Structure heart-failure-prediction/
```

│
├── data/
│   ├── raw/
│   │   └── heart.csv         # Original dataset
│   ├── processed/
│   │   ├── cleaned_dataset.csv       # Week 2  — cleaned, pre-ML
│   │   ├── ml_ready_dataset.csv      # Week 2 — encoded & scaled
│   │   └── final_modelling_dataset.csv    # Week 3 — refined, feature-engineered
│
├── notebooks/
│   ├── Week2_Feature_Engineering_Preprocessing.ipynb
│   └── Week3_Heart_Failure_Analysis.ipynb
│
├── reports/
│   ├── Business_Understanding_Report.pdf       # Week 2
│   ├── Data_Preprocessing_Report.pdf            # Week 2
│   └── Business_Insights_and_Recommendations_Report.docx  # Week 3
│
├── data_dictionary/
│   └── Data_Dictionary.md                      # Updated Week 3 with engineered features
│
├── images/
│   └── (exported chart images used in reports)
│
├── README.md
└── requirements.txt

```

## Dataset 📊 

https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

## Tools and Libraries📚  

The project was carried out using the following tools and libraries:

- Python
- Jupyter Notebook
- Pandas
- Scipy
- Matplotlib
- Seaborn
- Scikit-learn

## Week 2 - Feature Engineering & Data Preprocessing

### Methodology

The notebook follows this workflow:

1. Data Inspection — structure, data types, missing values, duplicates, summary statistics
2. Data Cleaning — disguised missing value treatment (0s in Cholesterol/RestingBP), duplicate check, data type validation
3. Feature Engineering — derived features (e.g., age grouping), column clarity review
4. Feature Encoding
Label Encoding for binary categoricals (Sex, ExerciseAngina)
One-Hot Encoding for nominal categoricals (ChestPainType, RestingECG, ST_Slope)
Outlier Detection & Treatment — boxplots, IQR method, capping applied to Cholesterol and Oldpeak
5. Feature Scaling — StandardScaler applied to numeric features
6. Feature Selection — correlation heatmap and Random Forest feature importance analysis

### Key Findings 📉 

- Identified and corrected disguised missing values in two clinical variables
- Selected encoding strategies appropriate to each categorical variable's structure (ordinal vs. nominal)
- Applied outlier capping to preserve sample size in a relatively small clinical dataset (918 records)
- Produced correlation and feature importance analysis to guide future model selection

### Visualizations 💹 
- Age distribution (histogram) 
- Boxplots for outlier detection (Age, RestingBP, Cholesterol, MaxHR, Oldpeak)
- Correlation heatmap
- Chest pain type vs. heart disease (count plot)
- Age vs. maximum heart rate by disease status (scatter plot)
- Feature importance ranking (bar chart)

### Conclusion

The preprocessing pipeline transformed 918 patient records into a machine-learning-ready dataset. The target variable is reasonably balanced (55.3% heart disease vs. 44.7% normal), so no resampling was needed. Numeric features (Age, RestingBP, Cholesterol, MaxHR, Oldpeak) were correctly standardized, confirmed by mean ≈ 0 and std ≈ 1 across all five.

## Week 3  — Advanced Data Analysis, Statistical Validation & Feature Engineering


### Methodology
Using the Week 2 cleaned dataset as the starting point, the Week 3 notebook follows this workflow:

* Advanced EDA — deeper distribution/skewness analysis, bivariate and multivariate relationships, correlation mapping (16+ visualizations)

* Statistical Hypothesis Testing — four tests, each with stated hypotheses, test statistics, p-values, and business interpretation

* Advanced Feature Engineering — three new features grounded in EDA and statistical findings

* Feature Evaluation & Selection — correlation and multicollinearity (VIF) analysis on original and engineered features

* Dataset Refinement — final modelling dataset produced based on evidence from steps above
* Statistical Analysis Summary

Four hypothesis tests were conducted on the cleaned dataset:

Test	Variables	Result	Outcome
* Welch's t-test	MaxHR × HeartDisease	t = 13.23, p < 0.000001	Reject H₀

* Chi-Square	ChestPainType × HeartDisease	χ² = 268.07, p < 0.000001	Reject H₀
* Mann-Whitney U	Cholesterol × ExerciseAngina	U = 112,811, p = 0.0039	Reject H₀ (weak practical effect)

* One-Way ANOVA	MaxHR × ST_Slope	F = 79.03, p < 0.000001	Reject H₀


**Standout finding:** Asymptomatic (no chest pain) patients showed a 79% disease rate vs. 14% for atypical angina — a counterintuitive result with direct clinical screening implications, since it suggests absence of chest pain should not be used to rule out heart disease risk.

### Feature Engineering 🔧 

Three new features were engineered and evaluated:

* AgeGroup — binned age bands (<40, 40-55, 55-70, 70+); disease rate rises from 34.4% to 70.9% across bands
* Chol_Age_Ratio — Cholesterol ÷ Age; removed from the final dataset due to high multicollinearity (VIF ≈ 20) and weak target correlation (-0.114)
* RiskScore — composite indicator combining ExerciseAngina, FastingBS, and above-median Oldpeak; the strongest predictor in the dataset (correlation 0.577 with target), outperforming every original feature


### Feature Evaluation & Selection ✒️ 

Correlation and Variance Inflation Factor (VIF) analysis were conducted on all original and engineered numeric features. Chol_Age_Ratio was flagged and removed for redundancy with its own source variables; all other features were retained with correlation and VIF evidence documented in the Feature Evaluation and Selection Summary.


## Key Outcomes 📈

* Validated four statistically significant clinical relationships through formal hypothesis testing.
* Identified a counterintuitive, clinically relevant finding around asymptomatic chest pain presentation.
* Engineered a composite Risk Score that outperforms all original features in target correlation.
* Identified and removed a multicollinear engineered feature before it could distort future modelling.
* Produced a refined, modelling-ready dataset for Week 4.


## Conclusion

Advanced statistical validation confirmed that several clinical variables — **MaxHR, ChestPainType, ST_Slope, and Oldpeak** — carry genuine, statistically significant relationships with heart disease presence, not just visual trends. The engineered **RiskScore** feature emerged as the single strongest predictor identified across the project so far. The resulting final_modelling_dataset.csv is ready to serve as the foundation for Week 4 machine learning model development.

## How to Run the Project

To run this project on your local machine:

1. Clone the repository.
2. Install the required Python libraries.
3. Open the Jupyter Notebook file.
4. Run the cells sequentially to reproduce the analysis.

## Author

Donald Nwachukwu   
  
AnalystLab
