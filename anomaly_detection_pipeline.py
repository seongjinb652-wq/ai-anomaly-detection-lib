# ============================================
# 📌 anomaly_detection_pipeline.py
# 목적: 이상 탐지 전체 파이프라인 실행 스크립트
# - 데이터 전처리 (KDD Cup)
# - 이상치 비율 축소
# - XGBoost 모델 학습 (이진/다중 분류)
# - Confusion Matrix 및 ROC-AUC 시각화
# - Feature Importance 분석
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. 데이터 로드 및 전처리
# -----------------------------
data_path = './data/kddcup.data.corrected'
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
             "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
             "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count",
             "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
             "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
             "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
             "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

df = pd.read_csv(data_path, header=None, names=col_names)

# 라벨 인코딩
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# 범주형 변수 원-핫 인코딩
cat_vars = ['protocol_type', 'service', 'flag', 'land', 'logged_in','is_host_login', 'is_guest_login']
cat_data = pd.get_dummies(df[cat_vars])

# 숫자형 변수
numeric_vars = list(set(df.columns) - set(cat_vars) - {'label'})
numeric_data = df[numeric_vars].copy()

# 결합
X = pd.concat([numeric_data, cat_data], axis=1)
y = df['label']

# train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# -----------------------------
# 2. 이상치 비율 축소 함수
# -----------------------------
def reduce_anomalies(df, pct_anomalies=0.01):
    labels = df['label'].copy()
    is_anomaly = labels != 0  # 여기서는 '0'을 normal로 가정
    num_normal = np.sum(~is_anomaly)
    num_anomalies = int(pct_anomalies * num_normal)
    all_anomalies = labels[labels != 0]
    anomalies_to_keep = np.random.choice(all_anomalies.index, size=num_anomalies, replace=False)
    anomalous_data = df.iloc[anomalies_to_keep].copy()
    normal_data = df[~is_anomaly].copy()
    new_df = pd.concat([normal_data, anomalous_data], axis=0)
    return new_df

# -----------------------------
# 3. XGBoost 모델 학습 (이진 분류 예시)
# -----------------------------
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 5,
    'eta': 0.1,
    'seed': 42
}
num_rounds = 50
evals = [(dtrain, 'train'), (dtest, 'test')]

model = xgb.train(params, dtrain, num_rounds, evals=evals)

# -----------------------------
# 4. 평가 및 시각화
# -----------------------------
preds = model.predict(dtest)
threshold = 0.5
pred_labels = (preds > threshold).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, pred_labels)
print("Confusion Matrix:\n", cm)

# ROC-AUC
auc = roc_auc_score(y_test, preds)
print("ROC-AUC:", auc)

# Accuracy
acc = accuracy_score(y_test, pred_labels)
print("Accuracy:", acc)

# ROC Curve 시각화
fpr, tpr, _ = roc_curve(y_test, preds)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Feature Importance
ax = xgb.plot_importance(model)
fig = ax.figure
fig.set_size_inches(10, 10)
plt.show()
