# ============================================
# 📌 preprocess_rtiot2022.py
# 목적: RT-IoT2022 데이터셋 전처리 및 학습/테스트 데이터셋 생성
# - 데이터 로드 및 컬럼 이름 지정 (RT-IoT2022 스키마 기반)
# - 정상/공격 라벨 인코딩(LabelEncoder)
# - 범주형 변수 원-핫 인코딩 처리
# - 숫자형 변수와 범주형 변수 결합
# - 학습/테스트 데이터셋 분할
# - 전처리된 데이터셋을 pickle 파일로 저장
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse
import pickle

# -----------------------------
# 1. Argument parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--pct_anomalies', default=.01, type=float,
                    help="비율에 맞게 이상치 샘플링 (정상 대비)")
parser.add_argument('--data_path', default='./data/RT-IoT2022.csv', type=str,
                    help="RT-IoT2022 데이터셋 경로")
args = parser.parse_args()

pct_anomalies = args.pct_anomalies
data_path = args.data_path

# -----------------------------
# 2. 데이터 로드
# -----------------------------
df = pd.read_csv(data_path)

# ⚠️ 주의: RT-IoT2022의 실제 컬럼명은 제공된 문서에 따라 수정 필요
# 예시로 'Label' 컬럼이 공격/정상 여부를 나타낸다고 가정
label_col = 'Label'

# -----------------------------
# 3. 라벨 인코딩
# -----------------------------
le = LabelEncoder()
df[label_col] = le.fit_transform(df[label_col])

# -----------------------------
# 4. 이상치 비율 축소 함수
# -----------------------------
def reduce_anomalies(df, pct_anomalies=.01, label_col='Label', normal_class=0):
    labels = df[label_col].copy()
    is_anomaly = labels != normal_class
    num_normal = np.sum(~is_anomaly)
    num_anomalies = int(pct_anomalies * num_normal)
    all_anomalies = labels[labels != normal_class]
    anomalies_to_keep = np.random.choice(all_anomalies.index, size=num_anomalies, replace=False)
    anomalous_data = df.iloc[anomalies_to_keep].copy()
    normal_data = df[~is_anomaly].copy()
    new_df = pd.concat([normal_data, anomalous_data], axis=0)
    return new_df

df = reduce_anomalies(df, pct_anomalies=pct_anomalies, label_col=label_col, normal_class=0)

# -----------------------------
# 5. 범주형/숫자형 변수 처리
# -----------------------------
# ⚠️ RT-IoT2022의 실제 범주형 변수 목록은 문서 확인 필요
cat_vars = ['Protocol', 'Service', 'Flag']  # 예시
cat_data = pd.get_dummies(df[cat_vars])

numeric_vars = list(set(df.columns) - set(cat_vars) - {label_col})
numeric_data = df[numeric_vars].copy()

numeric_cat_data = pd.concat([numeric_data, cat_data], axis=1)

# -----------------------------
# 6. 학습/테스트 분할
# -----------------------------
labels = df[label_col].copy()
integer_labels = le.transform(labels)

x_train, x_test, y_train, y_test = train_test_split(
    numeric_cat_data, integer_labels, test_size=.25, random_state=42
)

# -----------------------------
# 7. 저장
# -----------------------------
preprocessed_data = {
    'x_train': x_train,
    'y_train': y_train,
    'x_test': x_test,
    'y_test': y_test,
    'le': le
}

path = 'preprocessed_rtiot2022.pkl'
with open(path, 'wb') as out:
    pickle.dump(preprocessed_data, out)

print(f"✅ 전처리 완료: {path} 저장됨")
