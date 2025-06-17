#!/usr/bin/env python
# coding: utf-8

# In[26]:


import shutil
import sys
import os
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ultralytics import YOLO
import ultralytics
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import matplotlib.ticker as ticker
import warnings
import random
import seaborn as snsz
import csv
import datetime


# In[27]:


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from IPython.display import display

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from keras.preprocessing import image
import keras.backend as K
from tensorflow.keras.layers import *


import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[28]:


def create_num_to_label():
    with open("IP102/ip102_v1.1/num_label_reduced.txt") as f:
        num_to_label = {}
        for text in f.read().split("\n"):
            n,l = text.split(' ',1);
            num_to_label[n] = l.strip()

    return num_to_label

num_to_label = create_num_to_label()

def create_name_to_num():
    with open("IP102/ip102_v1.1/name_num_table.txt") as f:
        name_to_num = [] # class data
        for l in f.read().split("\n"):
            v = tuple(l.split())
            if v:
                if str(int(v[1])+1) in num_to_label.keys():
                    v = [v[0],str(int(v[1])+1)] # to fit the format in num_to_label.txt
                    name_to_num.append(v)
    
    return np.array(name_to_num)

name_to_num = create_name_to_num()


# In[29]:


original_DA_num_to_label = {23:"corn borer", 52:"blister beetle"} # original num_to_label

DA_label_to_num = {"corn borer":0, "blister beetle":1} # num_to_label for CGAN

def assign_augment_img(name_to_num):
    DA_path_to_num = []
    src_directory = "IP102/ip102_v1.1/images/"
    for name, num in zip(name_to_num[:, 0], name_to_num[:, 1]):
        num = int(num)
        if num in np.array(list(original_DA_num_to_label.keys())):
            DA_path_to_num.append([src_directory + name, DA_label_to_num[original_DA_num_to_label[num]]])

    return np.array(DA_path_to_num)


DA_path_to_num = assign_augment_img(name_to_num)
print(len(DA_path_to_num))


# In[61]:


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

train_size = 1500*2
batch_size = 64
img_size = 128
epoch_count = 300
noise_dim = 100 
n_class = 2
l_rate_g = 0.0001
l_rate_d = 0.0004
l_rate_c = 0.0004
b1 = 0.5
b2 = 0.999
r = 0.2
weight_decay = 1e-4
l2_reg = tf.keras.regularizers.l2(1e-4)



tags = list(DA_label_to_num.keys())

# Sample data
image_paths = DA_path_to_num[:,0][:train_size]
labels = DA_path_to_num[:,1][:train_size]

# Load images and labels
X_train = []
y_train = []

for path, label in zip(image_paths, labels):
    img = load_img(path, target_size=(img_size,img_size))
    img_array = img_to_array(img)
    X_train.append(img_array)
    y_train.append(label)

X_train = np.array(X_train).astype("float32") / 255.0

# Encode string labels into integers
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = np.array(y_train)

X_train = (X_train - 127.5) / 127.5
# dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

def data_generator():
    for x, y in zip(X_train, y_train):
        yield x, y

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=X_train.shape[1:], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)  # adjust if y is not scalar
    )
)

dataset = dataset.shuffle(buffer_size=1000).batch(batch_size, drop_remainder=True)


# In[62]:


print(dataset)
print("-------images---------")
print(f"Data Amount: {len(X_train)}")
print(f"Image Size : {len(X_train[0])} x {len(X_train[0,0])}")
print(f"Channel    : {len(X_train[0,0,0])}")
print(f"Shape      : {X_train.shape}")

print("-------labels---------")
print(f"Data Amount: {len(y_train)}")
print(f"Shape      : {y_train.shape}")  


# ### GACN with tensorflow

# In[89]:


bce_loss = tf.keras.losses.BinaryCrossentropy()
sparse_ce_loss = tf.keras.losses.SparseCategoricalCrossentropy()

def discriminator_loss(pred_real_source, pred_fake_source, pred_real_class, pred_fake_class, true_labels):
    ls_real = bce_loss(tf.ones_like(pred_real_source), pred_real_source)
    ls_fake = bce_loss(tf.zeros_like(pred_fake_source), pred_fake_source)
    ls = ls_real + ls_fake

    lc_real = bce_loss(tf.cast(true_labels, tf.float32), pred_real_class)
    lc_fake = bce_loss(tf.cast(true_labels, tf.float32), pred_fake_class)
    lc = lc_real + lc_fake

    return ls + lc

def generator_loss(pred_fake_source, pred_fake_class, true_labels, classifier_pred_real):
    ls_f = bce_loss(tf.ones_like(pred_fake_source), pred_fake_source)
    lc_f = bce_loss(tf.cast(true_labels, tf.float32), pred_fake_class)
    ltcr = bce_loss(tf.cast(true_labels, tf.float32), classifier_pred_real)
    return ls_f + lc_f + ltcr

def classifier_loss(pred_fake_class, pred_real_class, true_labels):
    global r
    y = tf.cast(true_labels, tf.float32)
    loss_fake = bce_loss(y, pred_fake_class)
    loss_real = bce_loss(y, pred_real_class)
    return r * loss_fake + loss_real


d_optimizer=Adam(learning_rate=l_rate_d, beta_1 =b1, beta_2 = b2)
g_optimizer=Adam(learning_rate=l_rate_g, beta_1 =b1, beta_2 = b2)
c_optimizer=Adam(learning_rate=l_rate_c, beta_1 =b1, beta_2 = b2)


# ### generator

# In[87]:


# LeakyReLU slope
alpha = 0.2

def build_generator():
    global alpha
    global noise_dim
    global n_class
    # Parameters
    RC = 512  # feature map depth after reshape

    z = Input(shape=(noise_dim,), name="noise_input")
    c = Input(shape=(n_class,), name="label_input")
    concatenate = Concatenate(name="concat_z_c")([z, c])  # shape: (noise_dim + n_class,)
    
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

g_model = build_generator()
g_model.summary()


# ### Discriminator

# In[90]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, BatchNormalization, Flatten, Dense

def build_discriminator(input_shape=(128, 128, 3), alpha=0.2):
    global n_class
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

    source_output = Dense(1, activation='sigmoid', name='source')(x) # fake or real
    class_output = Dense(1, activation='sigmoid', name='class')(x) # class label (0 or 1)

    model = tf.keras.Model(inputs=inputs, outputs=[source_output, class_output], name="Discriminator")
    return model

d_model = build_discriminator()
d_model.summary()


# ### Classifier ResNet18

# In[91]:


import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(x, filters, kernel_size=3, stride=1, downsample=False):
    global l2_reg
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

def build_resnet18():
    global n_class
    global l2_reg
    inputs = tf.keras.Input(shape=(128, 128, 3))

    x = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False, kernel_regularizer=l2_reg)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Conv2_x
    x = conv_block(x, 64)
    x = conv_block(x, 64)

    # Conv3_x
    x = conv_block(x, 128, stride=2, downsample=True)
    x = conv_block(x, 128)

    # Conv4_x
    x = conv_block(x, 256, stride=2, downsample=True)
    x = conv_block(x, 256)

    # Conv5_x
    x = conv_block(x, 512, stride=2, downsample=True)
    x = conv_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2_reg)(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet18")
    return model


c_model = build_resnet18()
c_model.summary()


# In[93]:


def show_samples(num_samples):
    global n_class
    global g_model
    fig, axes = plt.subplots(n_class,num_samples, figsize=(10,5)) 
    fig.tight_layout()

    for l in np.arange(n_class):
      random_latent_vectors = tf.random.normal(shape=(num_samples, noise_dim))
      one_hot_real_labels = tf.one_hot(np.full(num_samples, l), depth=n_class, dtype=tf.float32)
        
      gen_imgs = g_model({"noise_input": random_latent_vectors, "label_input": one_hot_real_labels}, training=True)
      for j in range(gen_imgs.shape[0]):
        img = image.array_to_img(gen_imgs[j], scale=True)
        axes[l,j].imshow(img)
        axes[l,j].yaxis.set_ticks([])
        axes[l,j].xaxis.set_ticks([])

        if j ==0:
          axes[l,j].set_ylabel(tags[l])
    plt.show()

def plot_loss(loss):
    print(loss)
    plt.figure(figsize=(10, 5))
    plt.plot(loss) 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training generator Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[94]:


@tf.function
def train_step(dataset):

    global g_model
    global d_model
    global c_model
    global r
    global n_class
    
    real_images, real_labels = dataset
    one_hot_real_labels = tf.one_hot(real_labels, depth=n_class, dtype=tf.float32)

    batch_size = tf.shape(real_images)[0]
    random_latent_vectors = tf.random.normal(shape=(batch_size, noise_dim))

    
    generated_images = g_model({"noise_input": random_latent_vectors, "label_input": one_hot_real_labels })

    
    with tf.GradientTape() as tape:
        pred_fake_s, pred_fake_c = d_model(generated_images, training=True)
        pred_real_s, pred_real_c = d_model(real_images, training=True)
        d_loss = discriminator_loss(pred_real_s, pred_fake_s, pred_real_c, pred_fake_c, real_labels)
    grads = tape.gradient(d_loss, d_model.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, d_model.trainable_variables))

    
    with tf.GradientTape() as tape:
        fake_images = g_model({"noise_input": random_latent_vectors, "label_input": one_hot_real_labels}, training=True)
        pred_fake_s, pred_fake_c = d_model(fake_images, training=True)
        classifier_pred_real = c_model(real_images, training=False)

        g_loss = generator_loss(pred_fake_s, pred_fake_c, real_labels, classifier_pred_real)
    grads = tape.gradient(g_loss, g_model.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, g_model.trainable_variables))


    with tf.GradientTape() as tape:
        pred_fake_cls = c_model(generated_images, training=True)
        pred_real_cls = c_model(real_images, training=True)
        c_loss = classifier_loss(pred_fake_cls, pred_real_cls, real_labels)

    grads = tape.gradient(c_loss, c_model.trainable_variables)
    c_optimizer.apply_gradients(zip(grads, c_model.trainable_variables))  

    return d_loss, g_loss, c_loss


# In[95]:


def train(dataset, epochs=epoch_count):
    best_loss = float('inf')
    patience = 50
    wait = 0
    g_loss_list = []
    d_loss_list = []
    c_loss_list = []
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        start = time.time()
        itern = 0

        for image_batch in tqdm(dataset):
            d_loss, g_loss, c_loss = train_step(image_batch)
            itern=itern+1
        if (epoch+1) % 10 == 0:
            show_samples(3)
            g_model.save(f"{dic}/generator.keras")
            d_model.save(f"{dic}/discriminator.keras") 
            c_model.save(f"{dic}/classifier.keras")

        g_loss_list.append(g_loss.numpy())
        d_loss_list.append(d_loss.numpy())
        c_loss_list.append(c_loss.numpy())
        plot_loss(g_loss_list)
        with open(f"{dic}/loss.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, itern, d_loss.numpy(), g_loss.numpy(), c_loss.numpy()])
            
            
        print (f'Epoch: {epoch} -- Generator Loss: {np.mean(g_loss_list)}, Discriminator Loss: {np.mean(d_loss_list)}, Classifier Loss: {np.mean(c_loss_list)}\n')
        print (f'Took {time.time()-start} seconds. \n\n') 

        # Early stopping check
        if np.mean(c_loss_list) < best_loss:
            best_loss = np.mean(c_loss_list)
            wait = 0
            print("New best classifier loss, resetting patience.")
        else:
            wait += 1
            print(f"No improvement. Patience: {wait}/{patience}")
        
        if wait >= patience:
            print("Early stopping triggered.")
            break


# In[ ]:


dic = f"GACN_result/{str(datetime.datetime.today())}"
os.mkdir(dic)
     
basic_info = {"train_size":train_size, "batch_size": batch_size, "image_size":img_size, "epoch": epoch_count, 
              "noise_dim":noise_dim, "n_class":n_class, "learning_rate_g": l_rate_g, "learning_rate_d":l_rate_d, 
              "learning_rate_c":l_rate_c, "beta_1":b1, "beta_2":b2,  "fake_coefficient":r, "weight_decay": weight_decay}

keys = basic_info.keys()
rows = basic_info.values()
with open(f"{dic}/basic_info.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(keys)
    writer.writerows([rows])

max_key_len = max(len(k) for k in basic_info.keys())
for k, v in basic_info.items():
    print(f"{k:<{max_key_len}} : {v}")
print("---------------------------------")

train(dataset, epochs=epoch_count)
g_model.save(f"{dic}/generator.keras")
d_model.save(f"{dic}/discriminator.keras") 
c_model.save(f"{dic}/classifier.keras")


# In[ ]:





# In[ ]:




