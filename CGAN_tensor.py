import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import csv

from cgan_utils import create_num_to_label, create_name_to_num, assign_augment_img
from cgan_models import build_generator, build_discriminator, build_resnet18
from cgan_train import train

from constants import (
    GAN_PARAMS, ORIGINAL_DA_NUM_TO_LABEL, DA_LABEL_TO_NUM, get_paths
)

# --- Data Preparation ---
paths = get_paths()
num_to_label = create_num_to_label(paths["num_label_file"])
name_to_num = create_name_to_num(paths["name_num_table"], num_to_label)
DA_path_to_num = assign_augment_img(name_to_num, ORIGINAL_DA_NUM_TO_LABEL, DA_LABEL_TO_NUM, paths["images_dir"])

image_paths = DA_path_to_num[:, 0][:GAN_PARAMS["TRAIN_SIZE"]]
labels = DA_path_to_num[:, 1][:GAN_PARAMS["TRAIN_SIZE"]]

X_train = []
y_train = []
for path, label in zip(image_paths, labels):
    img = image.load_img(path, target_size=(GAN_PARAMS["IMG_SIZE"], GAN_PARAMS["IMG_SIZE"]))
    img_array = image.img_to_array(img)
    X_train.append(img_array)
    y_train.append(label)
X_train = np.array(X_train).astype("float32") / 255.0
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = np.array(y_train)
X_train = (X_train - 127.5) / 127.5

def data_generator():
    for x, y in zip(X_train, y_train):
        yield x, y

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=X_train.shape[1:], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)
dataset = dataset.shuffle(buffer_size=1000).batch(GAN_PARAMS["BATCH_SIZE"], drop_remainder=True)

# --- Directory and Logging ---
dic = paths["CGAN_result_root"]
os.makedirs(dic, exist_ok=True)
basic_info = {
    "train_size": GAN_PARAMS["TRAIN_SIZE"], "batch_size": GAN_PARAMS["BATCH_SIZE"], "image_size": GAN_PARAMS["IMG_SIZE"], "epoch": GAN_PARAMS["EPOCH_COUNT"],
    "noise_dim": GAN_PARAMS["NOISE_DIM"], "n_class": GAN_PARAMS["N_CLASS"], "learning_rate_g": GAN_PARAMS["L_RATE_G"], "learning_rate_d": GAN_PARAMS["L_RATE_D"],
    "learning_rate_c": GAN_PARAMS["L_RATE_C"], "beta_1": GAN_PARAMS["B1"], "beta_2": GAN_PARAMS["B2"], "fake_coefficient": GAN_PARAMS["R"], "weight_decay": GAN_PARAMS["WEIGHT_DECAY"]
}
with open(f"{dic}/basic_info.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(basic_info.keys())
    writer.writerows([basic_info.values()])

# --- Training ---
g_model = build_generator(GAN_PARAMS["NOISE_DIM"], GAN_PARAMS["N_CLASS"])
d_model = build_discriminator(input_shape=(GAN_PARAMS["IMG_SIZE"], GAN_PARAMS["IMG_SIZE"], 3))
c_model = build_resnet18(input_shape=(GAN_PARAMS["IMG_SIZE"], GAN_PARAMS["IMG_SIZE"], 3), n_class=GAN_PARAMS["N_CLASS"])

train(
    dataset,
    epochs=GAN_PARAMS["EPOCH_COUNT"],
    g_model=g_model,
    d_model=d_model,
    c_model=c_model,
    noise_dim=GAN_PARAMS["NOISE_DIM"],
    n_class=GAN_PARAMS["N_CLASS"],
    d_optimizer=Adam(learning_rate=GAN_PARAMS["L_RATE_D"], beta_1=GAN_PARAMS["B1"], beta_2=GAN_PARAMS["B2"]),
    g_optimizer=Adam(learning_rate=GAN_PARAMS["L_RATE_G"], beta_1=GAN_PARAMS["B1"], beta_2=GAN_PARAMS["B2"]),
    c_optimizer=Adam(learning_rate=GAN_PARAMS["L_RATE_C"], beta_1=GAN_PARAMS["B1"], beta_2=GAN_PARAMS["B2"]),
    bce_loss=tf.keras.losses.BinaryCrossentropy(),
    r=GAN_PARAMS["R"]
)

g_model.save(f"{dic}/generator.keras")
d_model.save(f"{dic}/discriminator.keras")
c_model.save(f"{dic}/classifier.keras")