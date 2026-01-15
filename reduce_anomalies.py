# ============================================
# 📌 reduce_anomalies.py
# 목적: 데이터셋에서 정상 데이터 대비 이상치 비율을 줄여 균형 잡힌 데이터셋 생성
# - 입력: DataFrame (label 컬럼 포함)
# - 정상 데이터와 이상치 데이터 구분
# - 지정된 비율(pct_anomalies)에 맞게 이상치 샘플링
# - 정상 데이터와 샘플링된 이상치 데이터를 결합하여 새로운 DataFrame 반환
# - 실제 프로젝트에서는 이상치 탐지 모델 학습용 데이터 전처리에 활용
# ============================================

# def reduce_anomalies(df, pct_anomalies=.01):
#     labels = df['label'].copy()
#     is_anomaly = labels != 'normal.'
#     num_normal = np.sum(~is_anomaly)
#     num_anomalies = int(pct_anomalies * num_normal)
#     all_anomalies = labels[labels != 'normal.']
#     anomalies_to_keep = np.random.choice(all_anomalies.index, size=num_anomalies, replace=False)
#     anomalous_data = df.iloc[anomalies_to_keep].copy()
#     normal_data = df[~is_anomaly].copy()
#     new_df = pd.concat([normal_data, anomalous_data], axis=0)
#     return new_df

# ============================================
# 📌 reduce_anomalies.py
# 목적: 데이터셋에서 이상치(공격) 비율을 줄여 균형 잡힌 학습 데이터셋 생성
# - 정상 라벨과 공격 라벨을 구분
# - 정상 대비 일정 비율만큼 공격 샘플을 유지
# - 새로운 데이터프레임 반환
# ============================================

import numpy as np
import pandas as pd

def reduce_anomalies(df, pct_anomalies=0.01, label_col='label', normal_class='normal.'):
    """
    데이터셋에서 정상 대비 일정 비율의 이상치만 유지하는 함수
    
    Parameters
    ----------
    df : pandas.DataFrame
        원본 데이터셋
    pct_anomalies : float
        정상 데이터 대비 유지할 이상치 비율 (default=0.01)
    label_col : str
        라벨 컬럼명
    normal_class : str or int
        정상 데이터 라벨 값 (데이터셋에 따라 문자열 또는 숫자)
    
    Returns
    -------
    new_df : pandas.DataFrame
        이상치 비율이 줄어든 새로운 데이터셋
    """
    labels = df[label_col].copy()
    is_anomaly = labels != normal_class
    num_normal = np.sum(~is_anomaly)
    num_anomalies = int(pct_anomalies * num_normal)
    
    all_anomalies = labels[labels != normal_class]
    anomalies_to_keep = np.random.choice(all_anomalies.index, size=num_anomalies, replace=False)
    
    anomalous_data = df.loc[anomalies_to_keep].copy()
    normal_data = df.loc[~is_anomaly].copy()
    
    new_df = pd.concat([normal_data, anomalous_data], axis=0)
    return new_df


df = reduce_anomalies(df)

pd.DataFrame(df['label'].value_counts())
