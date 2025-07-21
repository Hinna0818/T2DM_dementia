## rf model construction
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
    df_mh = pd.read_excel("res_bal.xlsx")

    X = df_mh[['x1', 'x2', 'x3', 'x4', 'x5', 'x6' ,'x7' ,'x8', 'x9', 'x10', 
        'x11', 'x12', 'x13', 'x14', 'x15', 'cancer history', 'hypertension history', 
        'cholesterol usage', 'stroke_history', 'CVD_history', 'insulin_usage', 'APOE_status', 'sex', 
        'BMI_category', 'Family history of dementia', 'smoking', 'drinking']]
    y_ACD = df_mh['ACD']


    categorical_cols = [col for col in X.columns]

    # 预处理器
    preprocessor = ColumnTransformer([
    ('cat', 'passthrough', categorical_cols)
])

    # 处理 X
    X_processed = preprocessor.fit_transform(X)

    # 获取列名（sklearn >= 1.0）
    feature_names = preprocessor.get_feature_names_out()

    # 处理列名（去掉前缀 'num__' 或 'cat__'）
    feature_names = [name.split("__")[-1] for name in feature_names]

    # 包装成带列名的 DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    ## 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_ACD, 
        test_size=0.2,        
        random_state=23333, 
        shuffle=True     
    )


    # ## SMOTE过采样处理类别不平衡问题
    # # 初始化 SMOTE 对象
    # sm = SMOTE(random_state=42)

    # # 对训练集进行重采样
    # X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # # 使用 RFE 固定选10个特征
    # estimator = LogisticRegression(solver='liblinear', penalty='l2', random_state=42)
    # rfe = RFE(estimator=estimator, n_features_to_select=10, step=1)
    # rfe.fit(X_train_res, y_train_res)

    # # 变换数据
    # X_train_sel = rfe.transform(X_train_res)
    # X_test_sel = rfe.transform(X_test)
    
    # step 2: Optuna
    param = {
        'n_estimators': trial.suggest_int("n_estimators", 10, 20000, log=True),
        'max_depth': trial.suggest_int("max_depth", 2, 32),
        'min_samples_split': trial.suggest_int("min_samples_split", 2, 10),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 10),
        'class_weight' : trial.suggest_categorical('class_weight', ['balanced',None])
        
    }
    
    model = RandomForestClassifier(**param)       
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)
    y_pred_1_proba = model.predict_proba(X_train)[:, 1]
    tn, fp, fn, tp  = confusion_matrix(y_train, y_pred_proba.argmax(axis=1)).ravel()
    f2 = fbeta_score(y_train, y_pred, average='binary', beta=2)
    f0_5 = fbeta_score(y_train, y_pred, average='binary', beta=0.5)
    f1 = f1_score(y_train, y_pred, average='binary')
    precision, recall, _, _ = precision_recall_fscore_support(y_train, y_pred, average='binary')
    ROC_AUC = roc_auc_score(y_train, y_pred_1_proba, average='weighted')
    (precisions, recalls, _) = precision_recall_curve(y_train, y_pred_1_proba)
    aucpr = auc(recalls, precisions)
    AP = average_precision_score(y_train, y_pred_1_proba)
    
    brier = brier_score_loss(y_train, y_pred_1_proba)
    acc = accuracy_score(y_train, y_pred)
    fpr, tpr, thresholds = roc_curve(y_train, model.predict_proba(X_train)[:,1])
    print('{0} TN: {1} FP: {2} FN: {3} TP: {4} | Pre: {5:.3f} Rec: {6:.3f} F0.5: {7:.3f} F1: {8:.3f} F2: {9:.3f} AP: {10:.3f}| ROC_AUC: {11:.3f} AUCPR: {12:.3f} Brier: {13:.4f} ACC: {14:.4f}'
      .format('Model', tn, fp, fn, tp, precision, recall, f0_5, f1, f2, AP, ROC_AUC, aucpr, brier, acc))
    return f1


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