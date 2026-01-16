"""
Plotting GAN Training Losses
----------------------------
This script visualizes the training losses of the Generator and Discriminator 
during GAN training. It helps to monitor convergence and detect instability 
in adversarial learning.

네트워크 침입 탐지용 GAN 학습 손실 시각화
----------------------------
이 스크립트는 GAN 학습 과정에서 생성기와 판별기의 손실 곡선을 시각화합니다.
학습 수렴 여부와 적대적 학습의 불안정성을 모니터링하는 데 유용합니다.
"""
fig, ax = plt.subplots()
plt.plot(discriminator_loss, label='Discriminator')
plt.plot(gan_loss, label='Generator')
plt.title("Training Losses")
plt.legend()
