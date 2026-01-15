# ============================================
# 📌 plot_confusion_matrix.py
# 목적: 분류 모델의 Confusion Matrix를 시각화하는 함수 정의
# - confusion_matrix 결과를 matplotlib으로 시각화
# - 각 셀에 값(annotation) 표시
# - True label과 Predicted label 축을 포함
# - 모델 성능 평가 시 활용 가능
# - 실제 프로젝트에서는 분류 결과 분석 및 보고서 시각화에 확장
# ============================================

cm = confusion_matrix(true_labels, pred_labels)

print ('Confusion Matrix :')

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Greens):
    plt.figure(figsize=(10,10),)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    #tick_marks = np.arange(len(target_names))
    #plt.xticks(tick_marks, target_names, rotation=45)
    #plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(cm)
