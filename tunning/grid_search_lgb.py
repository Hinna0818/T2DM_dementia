import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Step 1: Load data
df_mh = pd.read_excel("res_bal.xlsx")

# Step 2: Features and label
X = df_mh[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
           'x11', 'x12', 'x13', 'x14', 'x15', 'cancer history', 'hypertension history',
           'cholesterol usage', 'stroke_history', 'CVD_history', 'insulin_usage', 'APOE_status', 'sex',
           'BMI_category', 'Family history of dementia', 'smoking', 'drinking']]
y = df_mh['ACD']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23333)

# Step 4: Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', 'passthrough', X.columns)
])

# Step 5: LightGBM pipeline
pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', lgb.LGBMClassifier(is_unbalance=True, verbose=-1))
])

# Step 6: Parameter grid
param_grid = {
    'clf__metric': ['auc'],
    'clf__random_state': [0, 42, 2021],
    'clf__n_estimators': [100, 500, 1000],
    'clf__reg_alpha': [0.001, 0.01, 0.1],
    'clf__reg_lambda': [0.001, 0.01, 0.1],
    'clf__colsample_bytree': [0.6, 0.8, 1.0],
    'clf__subsample': [0.6, 0.8, 1.0],
    'clf__learning_rate': [0.006, 0.01, 0.02],
    'clf__max_depth': [10, 20, 100],
    'clf__num_leaves': [15, 31, 63],
    'clf__min_child_samples': [5, 20],
    'clf__cat_smooth': [1, 20]
}

# Step 7: Cross-validation and grid search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipe, param_grid, scoring='f1', cv=cv, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Step 8: Results
print("\n✅ Best Parameters Found:")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\n⭐ Best F1 Score: {grid_search.best_score_:.4f}")

# Step 9: Test set evaluation
y_pred = grid_search.best_estimator_.predict(X_test)
print("\n📊 Test Set Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
