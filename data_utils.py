import os
import shutil
import numpy as np

def move_back_img(paths):
    for root, dirs, files in os.walk(paths["dataset_root"]):
        for file in files:
            if file.lower().endswith('.jpg'):
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(paths["images_dir"], file)
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

def move_img(train_val_test_pack, num_to_label, paths):
    src_directory = paths["images_dir"]
    for split_type, name_num_dic in train_val_test_pack.items():
        for name, num in zip(name_num_dic["name"], name_num_dic["num"]):
            dst_directory = f"{paths['dataset_root']}/{split_type}/{num_to_label[num]}"
            shutil.move(src_directory + name, dst_directory)