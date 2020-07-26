# Import modules

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Define Training Procedure

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(10, 10))

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='binary')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

seed = tf.random.normal(shape = [batch_size, 100])
def train_dcgan(gan, dataset, batch_size, num_features, epochs = 5):
    generator, discriminator = gan.layers
    for epoch in tqdm(range(epochs)):
        print("Epochs {}/{}".format(epochs+1, epochs))
        for X_batch in dataset:
            noise = tf.random.normal(shape = [batch_size, num_features])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis = 0)
            y1 = tf.constant([[0.]]* batch_size+[[1.]]*batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            y2 = tf.constant([[1.]]* batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
        display.clear_output(wait = True)
        generate_and_save_images(generator, epochs+1, seed)
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

#Train DCGAN

x_train_dcgan = x_train.reshape(-1, 28, 28, 1)*2. -1.

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(x_train_dcgan).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
train_dcgan(gan, dataset, batch_size, num_features, epochs=10)

#Generate Synthetic Images with DCGAN

noise = tf.random.normal(shape = [batch_size, num_features])
generated_images = generator(noise)
plot_utils.show(generated_images, 8)
