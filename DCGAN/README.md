## Deep Convolutional Generative Adversarial Network Overview
[DCGAN generator sturcture](https://miro.medium.com/max/1100/1*rdXKdyfNjorzP10ZA3yNmQ.webp) <br>


[DCGAN minmax loss function] (https://velog.velcdn.com/images%2Fhyebbly%2Fpost%2Fa6e590a2-92a6-4bde-8e10-70daf3103849%2Fimage.png)
<br>(D stands for discriminator, G stands for generator in this document)

###### DCGAN Guidelines
* Replace pooling layers with **strided convolution(D)** and **fractional-strided convolutions(G)**
*(Fractional-strided convolution shouldn't be confused as 'deconvolution')*
* Use **batchnorm** in both D and G
* Remove fully connected hidden layers for deeper architectures
* Use ReLU activation in G, and apply Tanh in final layer
* Use LeakyReLU ain the D for all layers

###### Training details
* Weights are initialized from zero-centered normal distribution , with 0.02 standard deviation
* Adam optimizer with 0.0002 learning rate. Momentum term *B1* 0.5 to stabilize the learning.

###### Research Papers
 [Unspuervised Representation Learning with Deep Convolutional Generative Adversarial Networks(2016)](https://arxiv.org/pdf/1511.06434.pdf)

 