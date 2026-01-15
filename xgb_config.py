# ============================================
# 📌 xgb_config.py
# 목적: XGBoost 모델 학습을 위한 하이퍼파라미터 설정 모음
# - num_rounds          : boosting 반복 횟수
# - max_depth, max_leaves: 트리 구조 제어
# - alpha, reg_lambda    : 정규화 파라미터
# - eta, learning_rate, gamma: 학습률 및 분할 제약
# - subsample            : 샘플링 비율
# - scale_pos_weight      : 클래스 불균형 보정
# - tree_method          : GPU 기반 학습 방식 지정
# - objective            : 이진 분류 목적 함수
# - verbose              : 학습 로그 출력 여부
# - 실제 프로젝트에서는 모델 학습/튜닝 시 import하여 사용
# ============================================

params = {
    'num_rounds':        10,
    'max_depth':         8,
    'max_leaves':        2**8,
    'alpha':             0.9,
    'eta':               0.1,
    'gamma':             0.1,
    'learning_rate':     0.1,
    'subsample':         1,
    'reg_lambda':        1,
    'scale_pos_weight':  2,
    'tree_method':       'gpu_hist',
    'n_gpus':            1,
    'objective':         'binary:logistic',
    'verbose':           True
}
