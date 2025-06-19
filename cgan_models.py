from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding,Concatenate, Flatten, Multiply,Conv2D,LeakyReLU, Dense, Reshape, Conv2DTranspose, BatchNormalization, ReLU

def build_generator(noise_dim, num_classes):
    noise = Input(shape=(noise_dim,))
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Embedding(num_classes, noise_dim)(label)
    label_embedding = Flatten()(label_embedding)

    x = Multiply()([noise, label_embedding])
    x = Dense(8 * 8 * 256, use_bias=False)(x)
    x = Reshape((8, 8, 256))(x)

    x = Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)  # 16x16
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)   # 32x32
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False)(x)   # 64x64
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(x)  # 128x128

    return Model([noise, label], x, name="Generator")

def build_discriminator(num_classes):
    image = Input(shape=(128, 128, 3))
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Embedding(num_classes, 128 * 128 * 3)(label)
    label_embedding = Flatten()(label_embedding)
    label_embedding = Reshape((128, 128, 3))(label_embedding)

    x = Concatenate(axis=-1)([image, label_embedding])

    x = Conv2D(64, 4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model([image, label], x, name="Discriminator")


