import tensorflow as tf
from tqdm import tqdm

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    return tf.reduce_mean(real_loss + fake_loss)

def generator_loss(fake_output):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))

def classifier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))

@tf.function
def train_step(g_model, d_model, c_model, batch, noise_dim, n_class, 
               g_optimizer, d_optimizer, c_optimizer, bce_loss, r):
    images, labels = batch
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, noise_dim])
    labels_onehot = tf.one_hot(labels, n_class)

    with tf.GradientTape(persistent=True) as tape:
        # Generate fake images
        generated_images = g_model([noise, labels_onehot], training=True)
        # Discriminator outputs
        real_output, real_class = d_model(images, training=True)
        fake_output, fake_class = d_model(generated_images, training=True)
        # Classifier outputs
        class_pred = c_model(images, training=True)
        # Losses
        d_loss = discriminator_loss(real_output, fake_output)
        g_loss = generator_loss(fake_output)
        c_loss = classifier_loss(labels, class_pred)
        total_g_loss = g_loss + r * c_loss

    d_gradients = tape.gradient(d_loss, d_model.trainable_variables)
    g_gradients = tape.gradient(total_g_loss, g_model.trainable_variables)
    c_gradients = tape.gradient(c_loss, c_model.trainable_variables)

    d_optimizer.apply_gradients(zip(d_gradients, d_model.trainable_variables))
    g_optimizer.apply_gradients(zip(g_gradients, g_model.trainable_variables))
    c_optimizer.apply_gradients(zip(c_gradients, c_model.trainable_variables))

    return d_loss, g_loss, c_loss

def train(dataset, epochs, g_model, d_model, c_model, noise_dim, n_class, 
          g_optimizer, d_optimizer, c_optimizer, bce_loss, r):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        progbar = tqdm(dataset, total=len(dataset))
        for batch in progbar:
            d_loss, g_loss, c_loss = train_step(
                g_model, d_model, c_model, batch, noise_dim, n_class, 
                g_optimizer, d_optimizer, c_optimizer, bce_loss, r
            )
            progbar.set_description(
                f"d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}, c_loss: {c_loss:.4f}"
            )