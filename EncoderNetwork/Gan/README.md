Anomaly Detection in Network Data using GANs

In the previous labs, we tried our hand at supervised and unsupervised anomaly detection using XGBoost and Deep Autoencoders on the KDD-99 network intrusion dataset.

We addressed the issue of unlabelled training data through the use of Deep Autoencoders in the second lab. However, unsupervised methods such as PCA and Autoencoders tend to be effective only on highly correlated data such as the KDD dataset, and these algorithms might also require the data to follow a Gaussian Distribution.

"Adversarial training (also called GAN for Generative Adversarial Networks), and the variations that are now being proposed, is the most interesting idea in the last 10 years in ML, in my opinion.". Yann LeCun, 2016.

What do GANs bring to the table and how are they different from Deep Autoencoders?

GANs are generative models that generate samples similar to the training dataset by learning the true data distribution. So instead of compressing the input into a latent space and classifying the test samples based on the reconstruction error, we actually train a classifier that outputs a probability score of a sample being Normal or Anomalous. As we will see later in the lab, this has positioned GANs as very attaractive unsupervised learning techniques.

GANs can be pretty tough to train and improving their stability is an active area of research today.
################
GAN을 이용한 네트워크 데이터의 이상 탐지¶

이전 연구에서는 KDD-99 네트워크 침입 데이터셋에서 XGBoost와 Deep Autoencoder를 이용한 감독 및 비감독 이상 감지를 시도했습니다.

두 번째 실험실에서는 딥 오토인코더를 사용하여 레이블이 없는 훈련 데이터 문제를 해결했습니다. 그러나 PCA나 오토인코더와 같은 비감독 방법은 KDD 데이터셋과 같이 상관관계가 높은 데이터에서만 효과적인 경향이 있으며, 이 알고리즘들은 데이터가 가우시안 분포를 따를 수도 있습니다.

"적대적 훈련(생성적 적대적 네트워크의 GAN)과 현재 제안되고 있는 변형들은 지난 10년간 머신러닝에서 가장 흥미로운 아이디어라고 생각합니다." 얀 르쿤, 2016년.

GAN은 무엇을 제공하며, 딥 오토인코더와 어떻게 다른가요?

GAN은 실제 데이터 분포를 학습하여 훈련 데이터셋과 유사한 샘플을 생성하는 생성형 모델입니다. 그래서 입력을 잠재 공간에 압축하고 재구성 오차에 따라 테스트 샘플을 분류하는 대신, 샘플이 정상(Normal) 또는 이상(Anomalous)일 확률 점수를 출력하는 분류기를 훈련합니다. 이후 연구실에서 보겠지만, 이로 인해 GAN은 매우 타입적이고 지도 없는 학습 기법으로 자리매김했습니다.

GAN은 훈련이 꽤 어려울 수 있으며, 그 안정성 향상은 오늘날 활발한 연구 분야입니다.
###################
References
Zenati, H., Foo, C., Lecouat, B., Manek, G. and Chandrasekhar, V. (2018). Efficient GAN-Based Anomaly Detection. [online] Arxiv.org. Available at: https://arxiv.org/abs/1802.06222
Ben Poole Alex Lamb Martin Arjovsky Olivier Mastropietro Vincent Dumoulin, Ishmael Belghazi and Aaron Courville. Adversarially learned inference. International Conference on Learning Representations, 2017.
Antonia Creswell, Tom White, Vincent Dumoulin, Kai Arulkumaran, Biswa Sengupta, and Anil A.Bharath. Generative adversarial networks: An overview. In the Proceedings of IEEE Signal Processing Magazine Special Issue on Deep Learning for Visual Understanding, accepted paper,2017.
Martin Renqiang Min Wei Cheng Cristian Lumezanu Daeki Cho Haifeng Chen Bo Zong, Qi Song.Deep autoencoding gaussian mixture model for unsupervised anomaly detection. International Conference on Learning Representations, 2018.
Shuangfei Zhai, Yu Cheng, Weining Lu, and Zhongfei Zhang. Deep structured energy based models for anomaly detection. International Conference on Machine Learning, pp. 1100-1109, 2016.
