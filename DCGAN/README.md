## DCGAN Overview

![DCGAN generator sturcture](https://miro.medium.com/max/1100/1*rdXKdyfNjorzP10ZA3yNmQ.webp) <br>
* DCGAN (Deep Convolutional Generative Adversarial Network) is a deep learning model that uses a pair of neural networks, a generator and a discriminator, to generate new images from a given input noise. The generator produces fake images that are fed into the discriminator, which then tries to distinguish between the real and fake images. The two networks are trained simultaneously, with the generator learning to produce images that can trick the discriminator and the discriminator learning to better distinguish between real and fake images. The result of this training process is a generator network that can produce high-quality, synthetic images that are indistinguishable from real images. DCGANs have been successfully applied in various domains, such as image synthesis, style transfer, and data augmentation.

![DCGAN minmax loss function](https://user-images.githubusercontent.com/86555104/216026302-d5c15dee-a735-4ab3-ad54-c644896152d7.png) <br>
* D(x) represents the probabilty that x came from the real data. G(z) maps the noise vector z to data-space which means D(G(z)) is a proability that D decides generator's image as a real image. Thus, D and G constantly competes while D tries to distinguish between real/generated fake data and G creates realistic image trying to decieve D. This competitive relationship is interpreted as minmaxgame: D tries to maximize logD(x), G tries to minimize the probability log(1-D(z)) that D correctly distinguish D(z) is fake.


###### (D stands for discriminator, G stands for generator in this document)

##### <ins>DCGAN Guidelines </ins>
* Replace pooling layers with **strided convolution(D)** and **fractional-strided convolutions(G)**
*(Fractional-strided convolution shouldn't be confused as 'deconvolution')*
* Use **batchnorm** in both D and G
* Remove fully connected hidden layers for deeper architectures
* Use ReLU activation in G, and apply Tanh in final layer
* Use LeakyReLU ain the D for all layers

##### <ins> Training details </ins>
* Weights are initialized from zero-centered normal distribution , with 0.02 standard deviation
* Adam optimizer with 0.0002 learning rate. Momentum term *B1* 0.5 to stabilize the learning.

##### <ins> Resources </ins>
 [Unspuervised Representation Learning with Deep Convolutional Generative Adversarial Networks(2016)](https://arxiv.org/pdf/1511.06434.pdf)
 [Pytorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

 
