# kaggle_porto

## Data Source
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

## File Description
 - a notebook that contains detail eda + modeling
 - a python script contain the whole training pipeline
    
## Description
This is a dataset containing auto insurance data. The goal is to use this dataset to train a model to predict if a drive will initiate a claim(1) or not(0). Training data has 595212 rows and 59 columns(including target and id). 
    
##  Preparation
 - check class balance for target
 - choose metric for evaluation
    * precision and recall
    * area under ROC curve
 - impute missing value
    * median for continuous variable
    * new class for categorical variable
 - reduce feature space
    * correlation matrix to find highly correlated feature or non-correlated features for continuous variable features
    * countplot w.r.t target for categorical features to find any irrelevant ones

## Preprocessing
 - Normalize continuous data to range of (0,1) to avoid large number effect
 - One-Hot encode categorical variables
 - Upsample minority class using SMOTE

## Modeling
 - Logistic Regression
 - Random Forest
 - Grid Searched Random Forest
 - K-FOLD Random Forest

