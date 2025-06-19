import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, LeakyReLU, Reshape, UpSampling2D, Conv2D, BatchNormalization, Dropout, Flatten, Add, ReLU, MaxPooling2D, GlobalAveragePooling2D)
from tensorflow.keras.models import Model

def build_generator(noise_dim, n_class, alpha=0.2):
    RC = 512
    z = Input(shape=(noise_dim,), name="noise_input")
    c = Input(shape=(n_class,), name="label_input")
    concatenate = tf.keras.layers.Concatenate(name="concat_z_c")([z, c])
    x = Dense(4 * 4 * RC, name="dense_map")(concatenate)
    x = LeakyReLU(alpha=alpha, name="dense_activation")(x)
    input_gen = Reshape((4, 4, RC), name="reshape_to_feature_map")(x)
    gen = UpSampling2D(size=(2, 2), interpolation="bilinear")(input_gen)
    gen = Conv2D(512, kernel_size=3, padding="same")(gen)
    gen = LeakyReLU(alpha)(gen)
    gen = UpSampling2D(size=(2, 2), interpolation="bilinear")(gen)
    gen = Conv2D(256, kernel_size=3, padding="same")(gen)
    gen = LeakyReLU(alpha)(gen)
    gen = UpSampling2D(size=(2, 2), interpolation="bilinear")(gen)
    gen = Conv2D(128, kernel_size=3, padding="same")(gen)
    gen = LeakyReLU(alpha)(gen)
    gen = UpSampling2D(size=(2, 2), interpolation="bilinear")(gen)
    gen = Conv2D(64, kernel_size=3, padding="same")(gen)
    gen = LeakyReLU(alpha)(gen)
    gen = UpSampling2D(size=(2, 2), interpolation="bilinear")(gen)
    gen = Conv2D(32, kernel_size=3, padding="same")(gen)
    gen = LeakyReLU(alpha)(gen)
    out = Conv2D(3, kernel_size=3, padding="same", activation="sigmoid")(gen)
    model = Model(inputs=[z, c], outputs=out, name="Generator")
    return model

def build_discriminator(input_shape=(128, 128, 3), alpha=0.2):
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(64, kernel_size=4, strides=2, padding="same")(inputs)
    x = LeakyReLU(alpha)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(512, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    source_output = Dense(1, activation='sigmoid', name='source')(x)
    class_output = Dense(1, activation='sigmoid', name='class')(x)
    model = tf.keras.Model(inputs=inputs, outputs=[source_output, class_output], name="Discriminator")
    return model

def conv_block(x, filters, kernel_size=3, stride=1, downsample=False, l2_reg=None):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False, kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=False, kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    if downsample:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, use_bias=False, kernel_regularizer=l2_reg)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def build_resnet18(input_shape=(128, 128, 3), n_class=2, l2_reg=None):
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False, kernel_regularizer=l2_reg)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    x = conv_block(x, 64, l2_reg=l2_reg)
    x = conv_block(x, 64, l2_reg=l2_reg)
    x = conv_block(x, 128, stride=2, downsample=True, l2_reg=l2_reg)
    x = conv_block(x, 128, l2_reg=l2_reg)
    x = conv_block(x, 256, stride=2, downsample=True, l2_reg=l2_reg)
    x = conv_block(x, 256, l2_reg=l2_reg)
    x = conv_block(x, 512, stride=2, downsample=True, l2_reg=l2_reg)
    x = conv_block(x, 512, l2_reg=l2_reg)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2_reg)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet18")
    return model