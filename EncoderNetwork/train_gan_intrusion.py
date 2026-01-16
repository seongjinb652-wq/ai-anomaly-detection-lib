"""
GAN Training Script for Network Intrusion Detection
---------------------------------------------------
This script trains a Generative Adversarial Network (GAN) to generate 
synthetic network packet data for anomaly detection research.

Key components:
- Generator: learns to produce realistic packet feature vectors
- Discriminator: distinguishes between real and generated samples
- Training loop: alternates between discriminator and generator updates

Dataset: Assumes preprocessed KDD-like intrusion detection dataset
Input dimension: 114 features per packet
---------------------------------------------------

네트워크 침입 탐지를 위한 GAN 학습 스크립트
---------------------------------------------------
이 스크립트는 생성적 적대 신경망(GAN)을 학습하여 
비정상 접근 탐지 연구에 활용할 수 있는 합성 네트워크 패킷 데이터를 생성합니다.

핵심 구성 요소:
- Generator: 실제와 유사한 패킷 특성 벡터를 생성
- Discriminator: 실제 데이터와 생성된 데이터를 구분
- Training loop: 판별기와 생성기를 번갈아 학습

데이터셋: KDD 계열 침입 탐지 데이터셋을 전처리하여 사용한다고 가정
입력 차원: 패킷당 114개 특성
"""
#Training the GAN
x_train, y_train, x_test, y_test = dataset['x_train'], dataset['y_train'],dataset['x_test'],dataset['y_test']

#Calculating the number of batches based on the batch size
batch_count = x_train.shape[0] // batch_size
pbar = tqdm(total=epochs * batch_count)
gan_loss = []
discriminator_loss = []

#Inititalizing the network
generator = get_generator(adam)
discriminator = get_discriminator(adam)
gan = get_gan_network(discriminator, generator, adam,input_dim=114)


for epoch in range(epochs):        
    for index in range(batch_count):        
        pbar.update(1)        
        # Creating a random set of input noise and images
        noise = np.random.normal(0, 1, size=[batch_size,114])
        
        # Generate fake samples
        generated_images = generator.predict_on_batch(noise)
        
        #Obtain a batch of normal network packets
        image_batch = x_train[index * batch_size: (index + 1) * batch_size]
            
        X = np.vstack((generated_images,image_batch))       
        y_dis = np.ones(2*batch_size) 
        y_dis[:batch_size] = 0

        # Train discriminator
        discriminator.trainable = True
        d_loss= discriminator.train_on_batch(X, y_dis)

        # Train generator
        noise = np.random.uniform(0, 1, size=[batch_size, 114])
        y_gen = np.ones(batch_size)
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y_gen)
        
        #Record the losses
        discriminator_loss.append(d_loss)
        gan_loss.append(g_loss)
        
    print("Epoch %d Batch %d/%d [D loss: %f] [G loss:%f]" % (epoch,index,batch_count, d_loss, g_loss))
