"""
GAN Test Set Predictions
------------------------
This script uses the trained discriminator to generate predictions 
on the test dataset. It processes the test set in batches and 
collects the results for further evaluation.

GAN 테스트셋 예측
------------------------
이 스크립트는 학습된 판별기를 이용하여 테스트 데이터셋에 대한 
예측을 수행합니다. 테스트셋을 배치 단위로 처리하며 
추후 평가를 위해 결과를 수집합니다.
"""

# Predictions on the test set

nr_batches_test = np.ceil(x_test.shape[0] // batch_size).astype(np.int32)

results =[]

for t in range(nr_batches_test +1):    
        ran_from = t * batch_size
        ran_to = (t + 1) * batch_size
        image_batch = x_test[ran_from:ran_to]             
        tmp_rslt = discriminator.predict(x=image_batch,batch_size=128,verbose=0)        
        results = np.append(results, tmp_rslt)   
