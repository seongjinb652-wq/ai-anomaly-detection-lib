# ============================================
# 📌 XGBoost_model_evaluation.py
# 목적: 분류 모델 성능 평가 및 모델 중요도 시각화
# - Confusion Matrix 시각화: Normal vs Anomaly 분류 결과 확인
# - ROC Curve 및 AUC 계산: 모델의 분류 성능 평가
# - XGBoost Feature Importance: 모델이 중요하게 판단한 특징 시각화
# - 실제 프로젝트에서는 모델 성능 검증 및 해석에 활용
# ============================================

# Confusion Matrix
results = confusion_matrix(true_labels, pred_labels) 

print ('Confusion Matrix :')

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


plot_confusion_matrix(results, ['Normal','Anomaly'])

# AUC 
fpr, tpr, thresholds = roc_curve(true_labels, pred_labels)
roc_auc = roc_auc_score(true_labels, pred_labels)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Model Inspection
ax = xgb.plot_importance(model)
fig = ax.figure
fig.set_size_inches(10, 10)


