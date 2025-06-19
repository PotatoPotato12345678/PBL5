import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import csv
from datetime import datetime

from data_utils import create_num_to_label, create_name_to_num
from cgan_utils import assign_augment_img
from cgan_models import build_generator, build_discriminator
from cgan_train import train

from constants import (get_paths, set_GAN_Result)

import constants

# --- Data Preparation ---
paths = get_paths()
num_to_label = create_num_to_label(paths)
name_to_num = create_name_to_num(paths, num_to_label)
DA_path_to_num = assign_augment_img(name_to_num, constants.ORIGINAL_DA_NUM_TO_LABEL, paths["images_dir"])

image_paths = DA_path_to_num[:, 0][:constants.GAN_PARAMS["TRAIN_SIZE"]]
labels = DA_path_to_num[:, 1][:constants.GAN_PARAMS["TRAIN_SIZE"]]

X_train = []
y_train = []
for path, label in zip(image_paths, labels):
    img = image.load_img(path, target_size=(constants.GAN_PARAMS["IMG_SIZE"], constants.GAN_PARAMS["IMG_SIZE"]))
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
dataset = dataset.shuffle(buffer_size=1000).batch(constants.GAN_PARAMS["BATCH_SIZE"], drop_remainder=True)
# --- Directory and Logging ---
dic = paths["CGAN_result_root"]

set_GAN_Result(f"{dic}/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}")
os.makedirs(constants.GAN_RESULT, exist_ok=True)
basic_info = {
    "train_size": constants.GAN_PARAMS["TRAIN_SIZE"], "batch_size": constants.GAN_PARAMS["BATCH_SIZE"], "image_size": constants.GAN_PARAMS["IMG_SIZE"], "epoch": constants.GAN_PARAMS["EPOCH_COUNT"],
    "noise_dim": constants.GAN_PARAMS["NOISE_DIM"], "n_class": constants.GAN_PARAMS["N_CLASS"], "learning_rate_g": constants.GAN_PARAMS["L_RATE_G"], "learning_rate_d": constants.GAN_PARAMS["L_RATE_D"],
    "learning_rate_c": constants.GAN_PARAMS["L_RATE_C"], "beta_1": constants.GAN_PARAMS["B1"], "beta_2": constants.GAN_PARAMS["B2"]
}
with open(f"{dic}/basic_info.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(basic_info.keys())
    writer.writerows([basic_info.values()])

# --- Training ---
g_model = build_generator(noise_dim=constants.GAN_PARAMS["NOISE_DIM"], num_classes=constants.GAN_PARAMS["N_CLASS"])
d_model = build_discriminator(num_classes=constants.GAN_PARAMS["N_CLASS"])

train(
    dataset,
    epochs=constants.GAN_PARAMS["EPOCH_COUNT"],
    g_model=g_model,
    d_model=d_model,
    noise_dim=constants.GAN_PARAMS["NOISE_DIM"],
    d_optimizer=Adam(learning_rate=constants.GAN_PARAMS["L_RATE_D"], beta_1=constants.GAN_PARAMS["B1"], beta_2=constants.GAN_PARAMS["B2"]),
    g_optimizer=Adam(learning_rate=constants.GAN_PARAMS["L_RATE_G"], beta_1=constants.GAN_PARAMS["B1"], beta_2=constants.GAN_PARAMS["B2"]),
)

g_model.save(f"{dic}/generator.keras")
d_model.save(f"{dic}/discriminator.keras")