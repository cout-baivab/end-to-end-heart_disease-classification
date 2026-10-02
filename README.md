End-to-End Heart Disease Classification

A machine learning pipeline that predicts whether a patient has heart disease from clinical attributes, built with scikit-learn. It covers the full workflow: data exploration, model comparison, hyperparameter tuning, and evaluation.

Problem

Given a patient's clinical measurements (age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, etc.), classify whether they have heart disease (target = 1) or not (target = 0).

In a screening context, missing a sick patient (false negative) is costlier than a false alarm, so recall is evaluated alongside accuracy.

Dataset

heart-disease.csv: tabular clinical data with a binary target. Based on the UCI Heart Disease (Cleveland) dataset.

Approach
1. Exploratory data analysis: class balance, feature distributions, correlations with the target
2. Baseline models: Logistic Regression, K-Nearest Neighbours, Random Forest
3. Hyperparameter tuning: RandomizedSearchCV / GridSearchCV with cross-validation
4. Evaluation: confusion matrix, classification report, precision, recall, F1, ROC curve and AUC
5. Feature importance: which attributes drive the prediction
