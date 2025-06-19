import numpy as np
import constants
import tensorflow as tf
import matplotlib.pyplot as plt


def assign_augment_img(name_to_num, ORIGINAL_DA_NUM_TO_LABEL, images_dir_path):
    DA_path_to_num = []
    for name, num in zip(name_to_num[:, 0], name_to_num[:, 1]):
        num = int(num)
        if num in ORIGINAL_DA_NUM_TO_LABEL.keys():
            DA_path_to_num.append([images_dir_path + name, num])
    return np.array(DA_path_to_num)

def show_generated_examples(g_model, epoch):
    num_classes = constants.GAN_PARAMS["N_CLASS"]
    noise_dim = constants.GAN_PARAMS["NOISE_DIM"]
    
    noise = tf.random.normal([num_classes, noise_dim])
    labels = tf.range(num_classes, dtype=tf.int32)
    labels = tf.reshape(labels, (-1, 1))

    generated_images = g_model([noise, labels], training=False)
    generated_images = (generated_images + 1) / 2.0  # scale [-1, 1] → [0, 1]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, num_classes, figsize=(num_classes * 2, 2))
    for i in range(num_classes):
        axes[i].imshow(generated_images[i])
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(constants.GAN_RESULT + f"/examples_{epoch}.png")
    plt.close()
