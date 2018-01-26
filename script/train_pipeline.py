import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,StratifiedKFold,KFold,GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer,OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as Pipeline_imb
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score,recall_score

import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")


def drop_record(df_train):
    mask = ((df_train['ps_ind_02_cat'] != df_train['ps_ind_02_cat'].max()) &
        (df_train['ps_ind_04_cat'] != df_train['ps_ind_04_cat'].max()) &
        (df_train['ps_car_01_cat'] != df_train['ps_car_01_cat'].max()) &
        (df_train['ps_car_02_cat'] != df_train['ps_car_02_cat'].max()) &
        (df_train['ps_car_09_cat'] != df_train['ps_car_09_cat'].max()))
    df_train = df_train[mask]
    return df_train


def df_tranform(df):
    df.drop(['id'], axis = 1, inplace = True)
    calc_column = [i for i in df.columns if 'calc' in i]
    df.drop(calc_column,axis = 1, inplace=True)
    summary = df.describe(include = 'all')
    null_columns = [i for i in summary.columns if summary[i]['count'] != 595212]
    for i in null_columns:
        if 'cat' in i:
            df_train[i].fillna((df_train[i].max()+1),inplace = True)
        else:
            df_train[i].fillna(df_train[i].median(), inplace = True)
    df.drop(['ps_ind_10_bin','ps_ind_13_bin','ps_car_10_cat'],axis = 1, inplace=True)
    df = df.rename(columns = {'ps_car_11_cat':'ps_car_11_num'})
    df = drop_record(df)
    bin_cat = [i for i in df.columns if ('bin' in i) | ('cat' in i)]
    for i in bin_cat:
        df[i] = df[i].astype('object')
    return df

def feature_selection(X,Y):
    reducer = RandomForestClassifier()
    reducer.fit(X,Y)
    ranking = [(i,j) for i,j in zip(X.columns,reducer.feature_importances_)]
    not_important = [i[0] for i in ranking if i[1] < .01]
    return not_important

def preprocessing(numeric_index,
                 categorical_index,
                 numeric_transformer,
                 categorical_transformer):
    pre_processor = FeatureUnion(transformer_list = [
                        #numeric
                        ('numeric_variables_processing', Pipeline(steps = [
                            ('selecting', FunctionTransformer(lambda data: data[:,numeric_index])),
                            ('scaling', numeric_transformer)])),
                        #categorical
                        ('categorical_variables_processing', Pipeline(steps = [
                            ('selecting', FunctionTransformer(lambda data: data[:,categorical_index])),
                            ('OneHot', categorical_transformer)]))])
    return pre_processor

def model_pipeline(preprocessor,
                   sampler,
                   model):
    return Pipeline_imb([('preprocessor',preprocessor),
                         ('sampler', sampler),
                         ('model',model)])

def AUROC(y_score, y_test):
    '''function to generate metrics given probabilites for the
       target class'''
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label = 1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr,lw=2,
             label='ROC curve (area = {:0.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    df_train = pd.read_csv('/Users/minheng/Documents/porto_seguro/train.csv',na_values="-1")
    df_train = df_tranform(df_train)
    Y = df_train['target']
    X = df_train.drop('target', axis = 1)
    not_important = feature_selection(X,Y)
    X.drop(not_important,axis = 1, inplace=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    n_column = X_train.dtypes != 'object'
    c_column = X_train.dtypes == 'object'
    pre = preprocessing(n_column,
                        c_column,
                        StandardScaler(),
                        OneHotEncoder())
    upsampler = SMOTE()
    model = LogisticRegression()
    estimator = model_pipeline(pre,
                               upsampler,
                               model)
    estimator.fit(X_train, Y_train)
    predict = estimator.predict(X_test)
    prob = estimator.predict_proba(X_test)
    AUROC(prob[:,1], Y_test)
