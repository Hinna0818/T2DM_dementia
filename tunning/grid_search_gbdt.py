import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")

# Step 1: 读取数据
df_mh = pd.read_excel("res_bal.xlsx")

# Step 2: 特征与标签
X = df_mh[['x1', 'x2', 'x3', 'x4', 'x5', 'x6' ,'x7' ,'x8', 'x9', 'x10', 
           'x11', 'x12', 'x13', 'x14', 'x15', 'cancer history', 'hypertension history', 
           'cholesterol usage', 'stroke_history', 'CVD_history', 'insulin_usage', 'APOE_status', 'sex', 
           'BMI_category', 'Family history of dementia', 'smoking', 'drinking']]
y = df_mh['ACD']

# Step 3: 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23333, shuffle=True
)

# Step 4: 预处理器（可按需更换为 scaler）
preprocessor = ColumnTransformer([
    ('cat', 'passthrough', X.columns)
])

# Step 5: 构建模型管道
gb_model = GradientBoostingClassifier(random_state=42)

pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', gb_model)
])

# Step 6: 设置参数网格（推荐控制搜索规模，初始调优用少量组合）
param_grid = {
    'clf__n_estimators': [100, 500, 1000],
    'clf__max_depth': [3, 5, 8],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 3, 5],
    'clf__min_weight_fraction_leaf': [0.0, 0.1, 0.3]
}

# Step 7: 网格搜索 + 交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='f1',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

# Step 8: 执行调参
grid_search.fit(X_train, y_train)

# Step 9: 输出最优结果
print("\n✅ Best Parameters Found:")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\n⭐ Best Cross-validated F1 Score: {grid_search.best_score_:.4f}")

# Step 10: 测试集表现
y_pred = grid_search.best_estimator_.predict(X_test)
print("\n📊 Test Set Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
