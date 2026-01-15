# ============================================
# 📌 xgb_multiclass_classifier.py
# 목적: XGBoost를 활용한 다중 클래스 분류 모델 학습 및 평가
# - objective: multi:softprob 설정으로 다중 클래스 확률 예측
# - num_class: 레이블 개수에 맞게 클래스 수 지정
# - dtrain, dtest, evals: 학습/테스트 데이터셋 구성
# - 모델 학습 후 예측값을 argmax로 변환하여 레이블 추출
# - Accuracy Score로 성능 평가
# - 실제 프로젝트에서는 다중 클래스 분류 문제 해결에 활용
# ============================================

num_labels = len(le.classes_)
params['objective'] = 'multi:softprob'
params['num_class'] = num_labels
print(params)

%%time 

dtrain =  ##SEE BINARY CLASSIFIER ##
dtest =  ##SEE BINARY CLASSIFIER ##
evals =  ##SEE BINARY CLASSIFIER ##
model =  ##SEE BINARY CLASSIFIER ##

preds = model.predict(dtest)

pred_labels = np.argmax(preds, axis=1)

pred_labels

true_labels = y_test

true_labels

print ('Accuracy Score :', accuracy_score(true_labels, pred_labels))
