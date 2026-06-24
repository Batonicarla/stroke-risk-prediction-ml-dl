# stroke-risk-prediction-ml-dl
# Predicting Stroke Risk Using Machine Learning and Deep Learning

## Project Overview

Stroke is one of the leading causes of death and long-term disability worldwide. Early identification of individuals at risk can help healthcare professionals implement preventive interventions and improve patient outcomes.

This project investigates the use of Machine Learning (ML) and Deep Learning (DL) techniques for predicting stroke risk using clinical and demographic data. Multiple traditional ML and neural network models were developed, trained, and evaluated to determine the most effective approach for stroke prediction.



## Objectives

The objectives of this project were:

* Build a complete preprocessing pipeline for healthcare data.
* Handle severe class imbalance using SMOTE and class-weighting techniques.
* Compare traditional machine learning models with deep learning architectures.
* Evaluate models using clinically meaningful metrics.
* Identify the most important factors contributing to stroke risk.



## Dataset

Dataset: Kaggle Healthcare Stroke Prediction Dataset
https://www.kaggle.com/code/rishabh057/healthcare-dataset-stroke-data
* Total records: 5,110 patients
* Features: 11 clinical and demographic attributes
* Target variable: Stroke (0 = No Stroke, 1 = Stroke)

### Features

* Gender
* Age
* Hypertension
* Heart Disease
* Ever Married
* Work Type
* Residence Type
* Average Glucose Level
* BMI
* Smoking Status
* Stroke (Target)



## Exploratory Data Analysis

Key findings from EDA:

* Age was the strongest predictor of stroke.
* Patients above 60 years accounted for most stroke cases.
* Higher glucose levels were associated with increased stroke risk.
* The dataset was highly imbalanced:

  * Non-Stroke: 95.13%
  * Stroke: 4.87%

Because of this imbalance, accuracy alone was not used as the primary evaluation metric.



## Data Preprocessing

The preprocessing pipeline included:

* Removal of unnecessary identifiers
* Median imputation for missing BMI values
* Label Encoding and One-Hot Encoding of categorical variables
* Feature scaling using StandardScaler
* Class balancing using SMOTE
* Stratified train/validation/test split



## Models Implemented

### Traditional Machine Learning

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Tuned XGBoost

### Deep Learning

5. Sequential Deep Neural Network
6. Functional API Deep Neural Network
7. Class-Weighted Deep Neural Network



## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

AUC-ROC was selected as the primary evaluation metric because it is more reliable for imbalanced datasets.



## Results

| Model               | Accuracy | Precision | Recall | F1-Score | AUC   |
| ------------------- | -------- | --------- | ------ | -------- | ----- |
| Logistic Regression | 79.3%    | 0.21      | 0.82   | 0.34     | 0.857 |
| Random Forest       | 94.6%    | 0.58      | 0.72   | 0.64     | 0.892 |
| XGBoost             | 95.1%    | 0.61      | 0.69   | 0.65     | 0.895 |
| Sequential DNN      | 93.8%    | 0.53      | 0.74   | 0.62     | 0.881 |
| Functional API DNN  | 94.9%    | 0.60      | 0.76   | 0.67     | 0.903 |
| DNN + Class Weights | 82.1%    | 0.24      | 0.87   | 0.38     | 0.870 |
| Tuned XGBoost       | 93.7%    | 0.52      | 0.78   | 0.62     | 0.901 |

### Best Model

The Functional API Deep Neural Network achieved the best overall performance:

* AUC = 0.903
* F1-Score = 0.67
* Recall = 0.76

The architecture processed numerical and categorical features separately before combining them, leading to improved generalization and reduced overfitting.



## Key Findings

* Age and average glucose level were the most important predictors.
* Deep learning slightly outperformed traditional machine learning approaches.
* Class imbalance significantly affected model performance.
* The Functional API DNN provided the best balance between precision and recall.



## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow
* Keras
* XGBoost
* Imbalanced-learn (SMOTE)



## Future Improvements

Potential improvements include:

* Transformer-based tabular models
* SHAP explainability analysis
* Probability calibration
* Larger real-world clinical datasets
* External validation on independent patient populations



## Author

**Carla Batoni**

Machine Learning Summative Project

24 June 2026

This project is intended for educational and research purposes.
