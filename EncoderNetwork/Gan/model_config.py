# model_config.py
"""
GAN Training Configuration
--------------------------
This configuration file defines the key hyperparameters for training 
the GAN model, including learning rate, batch size, number of epochs, 
and optimizer settings.

GAN 학습 설정
--------------------------
이 설정 파일은 GAN 모델 학습을 위한 주요 하이퍼파라미터를 정의합니다. 
학습률, 배치 크기, 에포크 수, 옵티마이저 설정 등이 포함됩니다.
"""

learning_rate = 0.00001
batch_size = 512
epochs = 10
adam = Adam(learning_rate = learning_rate,beta_1 = 0.5)
