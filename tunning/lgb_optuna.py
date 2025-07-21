## xgb model construction
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
    df_mh = pd.read_excel("res_new.xlsx")

    X = df_mh[['x1', 'x2', 'x3', 'x4', 'x5', 'x6' ,'x7' ,'x8', 'x9', 'x10', 
        'x11', 'x12', 'x13', 'x14', 'x15', 'cancer history', 'hypertension history', 
        'cholesterol usage', 'stroke_history', 'CVD_history', 'insulin_usage', 'APOE_status', 'sex', 'BMI_category',
         'Family history of dementia', 'smoking', 'drinking']]
    y_ACD = df_mh['ACD']


    categorical_cols = [col for col in X.columns]

    # 预处理器
    preprocessor = ColumnTransformer([
    ('cat', 'passthrough', categorical_cols)
])


    ## 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_ACD, 
        test_size=0.2,        
        random_state=23333, 
        stratify=y_ACD,
        shuffle=True     
    )

    def balance_dataset(X, y, pos_rep=3, random_state=42):
        # 合并数据
        data = X.copy()
        data['ACD'] = y.values

        # 正负类拆分
        pos = data[data['ACD'] == 1]
        neg = data[data['ACD'] == 0]

        # 正类复制
        pos_aug = pd.concat([pos] * pos_rep, ignore_index=True)

        # 负类下采样至与正类相同数量
        neg_sampled = neg.sample(n=len(pos_aug), random_state=random_state)

        # 合并打乱
        data_bal = pd.concat([pos_aug, neg_sampled], ignore_index=True).sample(frac=1, random_state=random_state)

        # 分离特征和标签
        X_bal = data_bal.drop(columns=['ACD']).reset_index(drop=True)
        y_bal = data_bal['ACD'].reset_index(drop=True)

        return X_bal, y_bal

    # 对训练集和测试集分别进行处理
    X_train, y_train = balance_dataset(X_train, y_train, pos_rep=6)
    X_test, y_test = balance_dataset(X_test, y_test, pos_rep=5)

    
    # step 2: Optuna
    param = {
        'metric': trial.suggest_categorical('metric', ['auc',""]),
        'random_state': trial.suggest_categorical('random_state' , [0, 42, 2021, 555]),
        'is_unbalance' : True,
        'n_estimators': trial.suggest_int('n_estimators', 1, 20000),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
        'num_leaves' : trial.suggest_int('num_leaves', 2, 1000), # num_leaves=1报错
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100),
        'verbosity': -1
    }
    
    model = lgb.LGBMClassifier(**param)       
    model.fit(X_train, y_train)

    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    y_pred_1_proba = model.predict_proba(X_test)[:, 1]
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
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    print('{0} TN: {1} FP: {2} FN: {3} TP: {4} | Pre: {5:.3f} Rec: {6:.3f} F0.5: {7:.3f} F1: {8:.3f} F2: {9:.3f} AP: {10:.3f}| ROC_AUC: {11:.3f} AUCPR: {12:.3f} Brier: {13:.4f} ACC: {14:.4f}'
      .format('Model', tn, fp, fn, tp, precision, recall, f0_5, f1, f2, AP, ROC_AUC, aucpr, brier, acc))
    return f1


if __name__ == "__main__":
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    study.optimize(objective, n_trials=300, timeout=1000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))