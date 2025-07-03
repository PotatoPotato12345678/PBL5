import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import constants
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import array_to_img
import os

def traditional_DA(img_paths, prms):
    augmented_images = []
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')

        transform = transforms.Compose([
            transforms.RandomRotation(degrees=prms['rotation_range']),
            transforms.ColorJitter(
                brightness=prms['brightness_range'],
                contrast=prms['contrast_range']
            ),
            transforms.RandomAffine(
                degrees=0,
                shear=prms['shear_range']
            ),
            transforms.ToTensor()
        ])

        torch.manual_seed(0)
        transformed_img = transform(img)

        np_img = transformed_img.permute(1, 2, 0).numpy()
        np_img = np.clip(np_img, 0, 1)
        augmented_images.append(np_img)
    return augmented_images

def generate_augmented_data(name_train, num_train, paths, fold_counter):
    name_train = np.array(name_train)
    num_train = np.array(num_train)
    orig_name_train, orig_num_train = [], []
    DA_name_train, DA_num_train = [], []

    key_list = list(constants.ORIGINAL_DA_NUM_TO_LABEL.keys())

    for label in np.unique(num_train):
        idxs = np.where(num_train == label)[0]

        if int(label) in constants.ORIGINAL_DA_NUM_TO_LABEL.keys():
            key_i = key_list.index(int(label))

            n = len(idxs)
            n_replace = n // 2
            replace_idxs = idxs[:n_replace]
            keep_idxs = idxs[n_replace:]
            img_paths = [os.path.join(paths["images_dir"], name_train[idx]) for idx in replace_idxs]
            if constants.DA_METHOD == "NON_AI_BASED":
                augmented_imgs = traditional_DA(img_paths, constants.AUGMENTATION_PARAMS)
            elif constants.DA_METHOD == "GAN_BASED":
                g_model = keras.models.load_model(constants.GENERATOR_MODEL_PATH)
                noise_dim = constants.GAN_PARAMS["NOISE_DIM"]

                noise = tf.random.normal([n_replace, noise_dim])
                labels = tf.constant(int(key_i), shape=(n_replace, 1), dtype=tf.int32)
                labels = tf.reshape(labels, [-1])
                
                augmented_imgs = g_model([noise, labels], training=True)
                augmented_imgs = (augmented_imgs + 1) / 2.0 


            aug_dir = os.path.join(paths["augmented_images_dir"], str(label))
            os.makedirs(aug_dir, exist_ok=True)
            for i, aug_img in enumerate(augmented_imgs):
                aug_name = f"aug_{constants.DA_METHOD}_{fold_counter}_{i}_{label}.jpg"
                aug_path = os.path.join(aug_dir, aug_name)
                array_to_img(aug_img).save(aug_path)
                DA_name_train.append(aug_name)
                DA_num_train.append(label)
            
            for idx in keep_idxs:
                orig_name_train.append(name_train[idx])
                orig_num_train.append(num_train[idx])
        else:
            for idx in idxs:
                orig_name_train.append(name_train[idx])
                orig_num_train.append(num_train[idx])
    return (
        np.array(orig_name_train), np.array(orig_num_train),
        np.array(DA_name_train), np.array(DA_num_train)
    )
