import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
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

# Step 3: 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23333, shuffle=True
)

# Step 4: 预处理器
preprocessor = ColumnTransformer([
    ('cat', 'passthrough', X.columns)
])

# Step 5: 模型构建
svm_model = SVC(probability=True, random_state=42)

pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', svm_model)
])

# Step 6: 参数网格（可根据训练速度适当扩展）
param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': ['scale', 'auto']  # 'rbf' 核使用 gamma
}

# Step 7: 网格搜索
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='f1',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

# Step 8: 模型拟合
grid_search.fit(X_train, y_train)

# Step 9: 输出结果
print("\n✅ Best Parameters Found:")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\n⭐ Best Cross-validated F1 Score: {grid_search.best_score_:.4f}")

# Step 10: 测试集评估
y_pred = grid_search.best_estimator_.predict(X_test)
print("\n📊 Test Set Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
