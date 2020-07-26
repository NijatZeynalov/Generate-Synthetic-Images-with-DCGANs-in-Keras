# Import modules

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Build the Generator Network for DCGAN

num_features = 100
generator = keras.models.Sequential([
   keras.layers.Dense(7**7**128, input_shape = [num_features]),
   keras.layers.Reshape([7, 7, 128]),
   keras.layers.BatchNormalization(),
   keras.layers.Conv2DTranspose(64, (5,5), (2,2), padding = 'same', activation = 'selu'),
   keras.layers.BatchNormalization(),
   keras.layers.Conv2DTranspose(1, (5,5), (2,2), padding = 'same', activation = 'tanh')
])

# Build the Discriminator Network for DCGAN
discriminator = keras.models.Sequential([
  keras.layers.Conv2D(64, (5,5), (2,2), padding = 'same', input_shape = [28, 28, 1]),
  keras.layers.LeakyReLU(0.2),
  keras.layers.Dropout(0.3),
  keras.layers.Conv2D(128, (5,5),(2,2), padding = 'same'),
   keras.layers.LeakyReLU(0.2),
  keras.layers.Dropout(0.3),
  keras.layers.Flatten(),
  keras.layers.Dense(1, activation = 'sigmoid')
])

# Compile the Deep Convolutional Generative Adversarial Network (DCGAN)
discriminator.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')
discriminator.trainable = False
gan = keras.models.Sequential([generator, discriminator])
gan.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')