import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import confusion_matrix
import constants
import re

def move_img(train_val_test_pack, num_to_label, paths):
    src_directory = paths["images_dir"]
    aug_src_directory = paths["augmented_images_dir"]
    for split_type, name_num_dic in train_val_test_pack.items():
        for name, num in zip(name_num_dic["name"], name_num_dic["num"]):
            dst_directory = f"{paths['dataset_root']}/{split_type}/{num_to_label[num]}"
            if split_type == "DA_train":
                # Augmented images are stored in subfolders by label
                src_file = os.path.join(aug_src_directory, str(num), name)
                dst_directory = f"{paths['dataset_root']}/train/{num_to_label[num]}"
            else:
                src_file = os.path.join(src_directory, name)
            if os.path.exists(src_file):
                os.makedirs(dst_directory, exist_ok=True)
                shutil.move(src_file, os.path.join(dst_directory, name))

def move_back_img(paths):
    images_dir = paths["images_dir"]
    aug_images_dir = paths["augmented_images_dir"]
    for root, dirs, files in os.walk(paths["dataset_root"]):
        for file in files:
            if file.lower().endswith('.jpg'):
                src_file_path = os.path.join(root, file)
                # Use regex to check if file is augmented
                if re.match(r'^aug_', file):
                    # Try to extract label from directory structure
                    try:
                        label = os.path.relpath(root, paths["dataset_root"]).split(os.sep)[1]
                    except IndexError:
                        label = None
                    if label is not None:
                        dst_dir = os.path.join(aug_images_dir, label)
                        os.makedirs(dst_dir, exist_ok=True)
                        dst_file_path = os.path.join(dst_dir, file)
                    else:
                        dst_file_path = os.path.join(aug_images_dir, file)
                else:
                    dst_file_path = os.path.join(images_dir, file)
                shutil.move(src_file_path, dst_file_path)

def create_num_to_label(paths):
    with open(paths["num_label_file"]) as f:
        num_to_label = {}
        for text in f.read().split("\n"):
            if text.strip():
                n, l = text.split(' ', 1)
                num_to_label[n] = l.strip()
    return num_to_label

def create_name_to_num(paths, num_to_label):
    with open(paths["name_num_table"]) as f:
        name_to_num = []
        for l in f.read().split("\n"):
            if l.strip():
                v = tuple(l.split())
                if str(int(v[1])+1) in num_to_label.keys():
                    v = [v[0], str(int(v[1])+1)]
                    name_to_num.append(v)
    return np.array(name_to_num)

def plot_class_dist(final_name_to_num, num_train, num_val, num_test, num_to_label, fold_count):
    num_list = {
        "Whole": final_name_to_num[:, 1],
        "Train": num_train,
        "Val": num_val,
        "Test": num_test
    }
    fig, ax = plt.subplots(2, 2, figsize=(18, 10))
    for i, (set_type, num) in enumerate(num_list.items()):
        dataset_count = Counter([num_to_label[single_num] for single_num in num])
        dataset_count = sorted(dataset_count.items(), key=lambda item: item[1], reverse=True)
        class_names, count_values = zip(*dataset_count)
        x = np.floor(i/2).astype(int)
        y = i % 2
        ax[x][y].bar(class_names, count_values)
        ax[x][y].set_title(f'{set_type} Data on Fold {fold_count}', fontsize=20, fontweight='bold')
        ax[x][y].set_xlabel('Class Name', fontsize=20)
        ax[x][y].set_ylabel('Data Size', fontsize=20)
        ax[x][y].xaxis.set_tick_params(rotation=90)
        max_count = max(count_values)
        n = int(np.floor(np.log10(max_count)))
        step = 10 ** n
        ax[x][y].yaxis.set_major_locator(ticker.MultipleLocator(step))
        ax[x][y].tick_params(axis='x', labelsize=5)
        ax[x][y].tick_params(axis='y', labelsize=20)
    fig.tight_layout(pad=3.0)
    fig.savefig(f'{constants.YOLO_RESULT}/class_dist_fold_{fold_count}.png')
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()