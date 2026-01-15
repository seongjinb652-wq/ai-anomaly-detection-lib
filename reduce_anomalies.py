# ============================================
# 📌 reduce_anomalies.py
# 목적: 데이터셋에서 정상 데이터 대비 이상치 비율을 줄여 균형 잡힌 데이터셋 생성
# - 입력: DataFrame (label 컬럼 포함)
# - 정상 데이터와 이상치 데이터 구분
# - 지정된 비율(pct_anomalies)에 맞게 이상치 샘플링
# - 정상 데이터와 샘플링된 이상치 데이터를 결합하여 새로운 DataFrame 반환
# - 실제 프로젝트에서는 이상치 탐지 모델 학습용 데이터 전처리에 활용
# ============================================

def reduce_anomalies(df, pct_anomalies=.01):
    labels = df['label'].copy()
    is_anomaly = labels != 'normal.'
    num_normal = np.sum(~is_anomaly)
    num_anomalies = int(pct_anomalies * num_normal)
    all_anomalies = labels[labels != 'normal.']
    anomalies_to_keep = np.random.choice(all_anomalies.index, size=num_anomalies, replace=False)
    anomalous_data = df.iloc[anomalies_to_keep].copy()
    normal_data = df[~is_anomaly].copy()
    new_df = pd.concat([normal_data, anomalous_data], axis=0)
    return new_df

df = reduce_anomalies(df)

pd.DataFrame(df['label'].value_counts())
