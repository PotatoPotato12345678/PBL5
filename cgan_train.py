import tensorflow as tf
from tqdm import tqdm
import constants
from cgan_utils import show_generated_examples


def discriminator_loss(real_output, fake_output):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return bce(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(g_model, d_model, batch, noise_dim, 
               g_optimizer, d_optimizer):
    images, labels = batch
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape(persistent=True) as tape:
        # Generate fake images
        generated_images = g_model([noise, labels], training=True)
        # Discriminator outputs
        real_output= d_model([images, labels],training=True)
        fake_output = d_model([generated_images, labels], training=True)
        # Losses
        d_loss = discriminator_loss(real_output, fake_output)
        g_loss = generator_loss(fake_output)

    d_gradients = tape.gradient(d_loss, d_model.trainable_variables)
    g_gradients = tape.gradient(g_loss, g_model.trainable_variables)

    d_optimizer.apply_gradients(zip(d_gradients, d_model.trainable_variables))
    g_optimizer.apply_gradients(zip(g_gradients, g_model.trainable_variables))

    return d_loss, g_loss

def train(dataset, epochs, g_model, d_model, noise_dim, 
          g_optimizer, d_optimizer):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        progbar = tqdm(dataset)
        for batch in progbar:
            d_loss, g_loss = train_step(
                g_model, d_model, batch, noise_dim, 
                g_optimizer, d_optimizer
            )
            progbar.set_description(
                f"d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}"
            )
        if (epoch + 1) % 1 == 0:
            show_generated_examples(g_model, epoch)

