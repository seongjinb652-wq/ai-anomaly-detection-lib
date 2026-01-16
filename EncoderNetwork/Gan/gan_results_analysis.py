"""
Analyzing GAN Discriminator Results
-----------------------------------
This script processes the prediction scores from the trained discriminator 
on the test dataset. It computes mean scores for normal and anomalous packets, 
and applies a thresholding strategy (lowest 1% score) to classify packets 
as normal (0) or anomalous (1).

GAN 판별기 결과 분석
-----------------------------------
이 스크립트는 학습된 판별기의 테스트 데이터셋 예측 점수를 처리합니다. 
정상 패킷과 비정상 패킷의 평균 점수를 계산하고, 
하위 1% 점수를 기준으로 임계값을 적용하여 
패킷을 정상(0) 또는 비정상(1)으로 분류합니다.
"""
pd.options.display.float_format = '{:20,.7f}'.format
results_df = pd.concat([pd.DataFrame(results),pd.DataFrame(y_test)], axis=1)
results_df.columns = ['results','y_test']
print ('Mean score for normal packets :', results_df.loc[results_df['y_test'] == 0, 'results'].mean() )
print ('Mean score for anomalous packets :', results_df.loc[results_df['y_test'] == 1, 'results'].mean())

#
#Obtaining the lowest 1% score
per = np.percentile(results,1)
y_pred = results.copy()
y_pred = np.array(y_pred)

#Thresholding based on the score
inds = (y_pred > per)
inds_comp = (y_pred <= per)
y_pred[inds] = 0
y_pred[inds_comp] = 1
