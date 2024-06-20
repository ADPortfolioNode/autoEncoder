import tensorflow as tf
import numpy as np
import traceback

import matplotlib.pyplot as plt

# Import MINST data
mnist = tf.keras.datasets.mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_image_train = tf.reshape(x_train, [-1, 28, 28, 1])
x_image_train = tf.cast(x_image_train, 'float32')

x_image_test = tf.reshape(x_test, [-1, 28, 28, 1])
x_image_test = tf.cast(x_image_test, 'float32')

print(x_train.shape)

flatten_layer = tf.keras.layers.Flatten()
x_train = flatten_layer(x_train)

print(x_train.shape)

learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10
global_step = tf.Variable(0)
total_batch = int(len(x_train) / batch_size)

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 128  # 2nd layer num features
encoding_layer = 32  # final encoding bottleneck features

# Building the encoder
encoder = tf.keras.Sequential([
    flatten_layer,
    tf.keras.layers.Dense(n_hidden_1, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(n_hidden_2, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(encoding_layer, activation=tf.nn.relu)
])

# Building the decoder
decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(n_hidden_2, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(n_hidden_1, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(n_input)
])

# AutoEncoder model
class AutoEncoder(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

model = AutoEncoder(encoder, decoder)
optimizer = tf.keras.optimizers.RMSprop(learning_rate)

def cost(targets, reconstruction):
    mse = tf.losses.MeanSquaredError()
    loss = mse(targets, reconstruction)
    return loss

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        reconstruction = model(inputs)
        loss_value = cost(targets, reconstruction)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

for epoch in range(training_epochs):
    for i in range(total_batch):
        x_inp = x_train[i * batch_size : (i + 1) * batch_size]
        if np.any(x_inp):
            loss_value, grads = grad(model, x_inp, x_inp)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(loss_value))
print("Optimization Finished!")

# Applying encode and decode over test set
encode_decode = model(flatten_layer(x_image_test[:examples_to_show]))

# Compare original images with their reconstructions
f, a = plt.subplots(2, examples_to_show, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(x_image_test[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    
plt.show()

print("Done.")

print(">>>>>>>>>>>>>>>>>>>>>end of autoencoder.py<<<<<<<<<<<<<<<<<<<<<<<<<")
