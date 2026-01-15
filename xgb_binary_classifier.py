# ============================================
# 📌 xgb_binary_classifier.py
# 목적: XGBoost를 활용한 이진 분류 모델 학습 및 평가
# - DMatrix로 학습/테스트 데이터셋 구성
# - 지정된 파라미터와 반복 횟수(num_rounds)로 모델 학습
# - 예측값을 threshold 기준으로 레이블 변환
# - ROC-AUC 및 Accuracy Score 계산으로 성능 평가
# - 실제 프로젝트에서는 이상 탐지, 스팸 필터링 등 이진 분류 문제 해결에 활용
# ============================================

x_train.head()

y_train[0:100]

%%time 

dtrain = xgb.DMatrix(x_train, label=binary_y_train)
dtest = xgb.DMatrix(x_test, label=binary_y_test)
evals = [(dtest, 'test',), (dtrain, 'train')]

num_rounds = params['num_rounds']

model = xgb.train(params, dtrain, num_rounds, evals=evals)

#!nvidia-smi

threshold = .5
true_labels = binary_y_test.astype(int)
true_labels.sum()

# make predictions on the test set using our trained model
preds = model.predict(dtest)
print(preds)

pred_labels = (preds > threshold).astype(int)
print(pred_labels)

pred_labels.sum()

# compute the auc
auc = roc_auc_score(true_labels, preds)
print(auc)

print ('Accuracy:', accuracy_score(true_labels, pred_labels))
