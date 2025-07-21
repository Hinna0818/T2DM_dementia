## svm model construction
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
# Sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.linear_model import LinearRegression, BayesianRidge, ridge_regression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    precision_recall_curve,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
    roc_curve,
    brier_score_loss,
    accuracy_score
)
import shap
import pyreadstat
from sklearn.compose import make_column_selector as selector
import copy
import lightgbm as lgb
from sklearn.model_selection import cross_validate
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC


def objective(trial: optuna.Trial) -> float:
    minmaxnorm = False
    SMOTE_Process = False
    SHAP_Analysis = False
    ALE_Analysis = False
    print('minmaxnorm: ', minmaxnorm)
    print('SMOTE_Process: ', SMOTE_Process)
    print('SHAP_Analysis: ', SHAP_Analysis)
    print('ALE_Analysis: ', ALE_Analysis)

    ## 导入数据
    df = pd.read_excel("final_data.xlsx")
    df['PRS'].value_counts()

    df_mh = df[df['PRS'].isin([2, 3])]

    X = df_mh[['x1', 'x2', 'x3', 'x4', 'x5', 'x6' ,'x7' ,'x8', 'x9', 'x10', 
        'x11', 'x12', 'x13', 'x14', 'x15', 'cancer_history', 'hypertension_history', 
        'choloresteral_history', 'Apoe', 'age', 'BMI', 'dementia_family_history']]
    y_ACD = df_mh['ACD']


    # 连续变量
    continuous_cols = ['age', 'BMI']
    categorical_cols = [col for col in X.columns if col not in continuous_cols]

    # 预处理器
    preprocessor = ColumnTransformer([
    ('num', StandardScaler(), continuous_cols),
    ('cat', 'passthrough', categorical_cols)
    ])

    # 处理 X
    X_processed = preprocessor.fit_transform(X)

    ## 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_ACD, 
        test_size=0.2,        
        random_state=42, 
        stratify=y_ACD     
    )


    ## SMOTE过采样处理类别不平衡问题
    # 初始化 SMOTE 对象
    sm = SMOTE(random_state=42)

    # 对训练集进行重采样
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # 使用 RFE 固定选10个特征
    estimator = LogisticRegression(solver='liblinear', penalty='l2', random_state=42)
    rfe = RFE(estimator=estimator, n_features_to_select=10, step=1)
    rfe.fit(X_train_res, y_train_res)

    # 变换数据
    X_train_sel = rfe.transform(X_train_res)
    X_test_sel = rfe.transform(X_test)
    
    # step 2: Optuna
    param = {
    'C': trial.suggest_float("C", 1e-3, 1e3, log=True),
    'kernel': trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
    'gamma': trial.suggest_categorical("gamma", ["scale", "auto"]),
    'degree': trial.suggest_int("degree", 2, 5) if trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]) == "poly" else 3,
    'probability': True,
    'shrinking': trial.suggest_categorical("shrinking", [True, False]),
    'tol': trial.suggest_float("tol", 1e-5, 1e-1, log=True)
}

    model = SVC(**param)     
    model.fit(X_train_sel, y_train_res)
    
    y_pred = model.predict(X_test_sel)
    y_pred_proba = model.predict_proba(X_test_sel)
    y_pred_1_proba = model.predict_proba(X_test_sel)[:, 1]
    tn, fp, fn, tp  = confusion_matrix(y_test, y_pred_proba.argmax(axis=1)).ravel()
    f2 = fbeta_score(y_test, y_pred, average='binary', beta=2)
    f0_5 = fbeta_score(y_test, y_pred, average='binary', beta=0.5)
    f1 = f1_score(y_test, y_pred, average='binary')
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    ROC_AUC = roc_auc_score(y_test, y_pred_1_proba, average='weighted')
    (precisions, recalls, _) = precision_recall_curve(y_test, y_pred_1_proba)
    aucpr = auc(recalls, precisions)
    AP = average_precision_score(y_test, y_pred_1_proba)
    
    brier = brier_score_loss(y_test, y_pred_1_proba)
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_sel)[:,1])
    print('{0} TN: {1} FP: {2} FN: {3} TP: {4} | Pre: {5:.3f} Rec: {6:.3f} F0.5: {7:.3f} F1: {8:.3f} F2: {9:.3f} AP: {10:.3f}| ROC_AUC: {11:.3f} AUCPR: {12:.3f} Brier: {13:.4f} ACC: {14:.4f}'
      .format('Model', tn, fp, fn, tp, precision, recall, f0_5, f1, f2, AP, ROC_AUC, aucpr, brier, acc))
    return ROC_AUC


if __name__ == "__main__":
    
  
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    study.optimize(objective, n_trials=500, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))