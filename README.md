# Generate Synthetic Images with DCGANs in Keras
In this hands-on project, I learned about Generative Adversarial Networks (GANs) and built a Deep Convolutional GAN (DCGAN) with Keras to generate images of fashionable clothes.  I used the Keras Sequential API with Tensorflow 2 as the backend. 

<a href="https://user-images.githubusercontent.com/31247506/88457521-fd16da80-ce8f-11ea-8fd5-84a054b634a5.png"><img src="https://user-images.githubusercontent.com/31247506/88457521-fd16da80-ce8f-11ea-8fd5-84a054b634a5.png" title="made at imgflip.com"/></a>


In our GAN setup,  we want to be able to sample from a complex, high-dimensional training distribution of the Fashion MNIST images. However, there is no direct way to sample from this distribution. The solution is to sample from a simpler distribution, such as Gaussian noise. We want the model to use the power of neural networks to learn a transformation from the simple distribution directly to the training distribution that we care about. 

__The GAN consists of two adversarial players: a discriminator and a generator. We’re going to train the two players jointly in a minimax game theoretic formulation.__

<a href="https://user-images.githubusercontent.com/31247506/88457537-2df70f80-ce90-11ea-9fad-7ca633c1558d.gif"><img src="https://user-images.githubusercontent.com/31247506/88457537-2df70f80-ce90-11ea-9fad-7ca633c1558d.gif" title="made at imgflip.com"/></a>


In this project, I focused on two learning objectives:

1. Understand Deep Convolutional Generative Adversarial Networks (DCGANs and GANs)

2. Design and train DCGANs using the Keras API in Python
