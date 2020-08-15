'''
  Author       : Bao Jiarong
  Creation Date: 2020-08-12
  email        : bao.salirong@gmail.com
  Task         : Denoise Autoencoder based on Keras Model
'''

import tensorflow as tf
#==========================Conv_AE based on Keras Model=========================
class Conv_ae(tf.keras.Model):
    def __init__(self, latent = 100, units = 32):
        super(Conv_ae, self).__init__()

        # Encoder
        self.conv1  = tf.keras.layers.Conv2D(filters = units * 2,kernel_size=(3,3),strides=(2,2),padding = 'valid',activation = "relu")
        self.conv2  = tf.keras.layers.Conv2D(filters = units * 4,kernel_size=(3,3),strides=(2,2),padding = 'valid',activation = "relu")
        self.flatten= tf.keras.layers.Flatten()

        # Latent
        self.la_dense= tf.keras.layers.Dense(units = latent, activation="relu")

        # Decoder
        self.dense1  = tf.keras.layers.Dense(units = 11*11*units * 4, activation = "relu")
        self.reshape = tf.keras.layers.Reshape((11,11,units * 4), name = "de_main_out")
        self.de_conv1= tf.keras.layers.Conv2DTranspose(filters = units * 2 , kernel_size=(3,3), strides=(2,2), padding='valid')
        self.de_conv2= tf.keras.layers.Conv2DTranspose(filters = 3 , kernel_size=(4,4), strides=(2,2), padding='valid',activation = "sigmoid")

    def call(self, inputs):
        x = inputs              #;print(x.shape)
        z = self.conv1(x)               #;print(z.shape)
        y = self.conv2(z)               #;print(y.shape)
        x = self.flatten(y)                 #;print(x.shape)
        x = self.la_dense(x)                #;print(x.shape)
        x = self.dense1(x)              #;print(x.shape)
        x = self.reshape(x) + y                 #;print(x.shape)
        x = self.de_conv1(x) + z                #;print(x.shape)
        x = self.de_conv2(x)                #;print(x.shape)
        return x

#------------------------------------------------------------------------------
def Conv_AE(input_shape, latent, units):
    model = Conv_ae(latent = latent, units = units)
    model.build(input_shape = input_shape)
    return model
