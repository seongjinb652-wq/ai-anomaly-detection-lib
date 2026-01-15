# ============================================
# 📌 convert_label_to_binary.py
# 목적: KDD Cup 데이터셋 전처리 (이진 분류용)
# - 데이터 로드 및 컬럼 이름 지정
# - 정상(normal.)을 0, 공격을 1로 변환. 이부분을 개발 목적에 따라 바꾸세용
# - 범주형 변수 원-핫 인코딩 처리
# - 숫자형 변수와 범주형 변수 결합
# - 학습/테스트 데이터셋 분할
# - 전처리된 데이터셋을 pickle 파일로 저장
# ============================================

# convert labels to binary (normal=0, attack=1)
def convert_label_to_binary(label_encoder, labels):
    normal_idx = np.where(label_encoder.classes_ == 'normal.')[0][0]
    my_labels = labels.copy()
    my_labels[my_labels != normal_idx] = 1 
    my_labels[my_labels == normal_idx] = 0
    return my_labels

binary_y_train = convert_label_to_binary(le, y_train)
binary_y_test = convert_label_to_binary(le, y_test)

print('Number of anomalies in y_train: ', binary_y_train.sum())
print('Number of anomalies in y_test:  ', binary_y_test.sum())
