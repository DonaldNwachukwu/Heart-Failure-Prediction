# Heart Failure Prediction — Feature Engineering & Data Preprocessing


##  📌 Project Overview

This project prepares the Heart Failure Prediction Dataset for machine learning by applying a full preprocessing pipeline — data cleaning, feature engineering, encoding, scaling, outlier treatment, and feature selection. 
  
## Business Problem

Heart failure occurs when the heart cannot pump enough blood to meet the body's needs. It’s usually triggered by other health conditions affecting the heart muscles, including coronary artery disease, heart attack, high blood pressure, heart valve disorders, congenital defects, cardiomyopathy, myocarditis, diabetes, and pulmonary hypertension.

While heart failure is often caused by underlying health conditions, each individual’s health risks are unique. Many factors like age, family history, and lifestyle can contribute, but early detection with advanced diagnostic tools can help manage and reduce the likelihood of serious damage. 

Therefore, a heart failure prediction model would be effective in detecting the likelihood or the presence of heart failure in individuals. 


## Objective

The objective is to predict the likelihood of a patient developing heart disease based on 11 clinical and diagnostic features, producing a dataset that is clean, well-structured, and ready for predictive modelling.

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

## Methodology

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

## Key Findings 📉 

📈 Key Outcomes
- Identified and corrected disguised missing values in two clinical variables
- Selected encoding strategies appropriate to each categorical variable's structure (ordinal vs. nominal)
- Applied outlier capping to preserve sample size in a relatively small clinical dataset (918 records)
- Produced correlation and feature importance analysis to guide future model selection

## Visualizations 💹 
- Age distribution (histogram) 
- Boxplots for outlier detection (Age, RestingBP, Cholesterol, MaxHR, Oldpeak)
- Correlation heatmap
- Chest pain type vs. heart disease (count plot)
- Age vs. maximum heart rate by disease status (scatter plot)
- Feature importance ranking (bar chart)

## Conclusion

This project provided a clear view of the factors associated with employee attrition. The findings can help organizations better understand workforce behavior and make informed decisions to improve employee retention.

## Recommendations

Based on the analysis, the following recommendations are suggested:

- Improve employee engagement and job satisfaction.
- Review compensation structures to ensure competitiveness.
- Strengthen workplace support and communication.
- Monitor employees who may be at higher risk of leaving.

## How to Run the Project

To run this project on your local machine:

1. Clone the repository.
2. Install the required Python libraries.
3. Open the Jupyter Notebook file.
4. Run the cells sequentially to reproduce the analysis.

## Author

Donald Nwachukwu

AnalystLab Africa
