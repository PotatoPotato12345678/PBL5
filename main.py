import os
import numpy as np
import random
import datetime
import csv
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img
from augmentation import traditional_DA
import constants
from constants import get_paths, get_params
from data_utils import move_back_img, create_num_to_label, create_name_to_num, move_img,plot_class_dist, plot_confusion_matrix
from yolo_utils import yolo_classify

OPTION_num = None

def prepare_data(params, paths):
    if params["move_back_flag"]:
        move_back_img(paths)
    num_to_label = create_num_to_label(paths)
    name_to_num = create_name_to_num(paths, num_to_label)
    if params["quick_demo_flag"]:
        if params["method_selection"] == 'down_sampling':
            grouped = {}
            for name, num in zip(name_to_num[:, 0], name_to_num[:, 1]):
                grouped.setdefault(num, []).append(name)

            # min_size = min(len(names) for names in grouped.values())
            # accept_min = int(np.floor(min_size / (3 * params["k_folds"])))
            # if params["down_sampling_size"] > accept_min:
            #     params["down_sampling_size"] = accept_min

            random.seed(params["random_seed"])
            reduced_dataset_name = []
            reduced_dataset_num = []

            for num, names in grouped.items():
                extend_size = int(params["down_sampling_size"] * (5/3))
                reduced_dataset_name.extend(random.sample(names, extend_size))
                reduced_dataset_num.extend([num] * extend_size)
            final_name_to_num = np.array(list(zip(reduced_dataset_name, reduced_dataset_num)))
        elif params["method_selection"] == 'stratified':
            _, name_reduced_data, _, num_reduced_data = train_test_split(
                name_to_num[:, 0],
                name_to_num[:, 1],
                stratify=name_to_num[:, 1],
                test_size=params["stratified_size"],
                random_state=params["random_seed"]
            )
            final_name_to_num = np.array(list(zip(name_reduced_data, num_reduced_data)))
    else:
        final_name_to_num = name_to_num.copy()
    return final_name_to_num, num_to_label

def split_fold_data(final_name_to_num, train_index, test_index, params):
    test_dataset = np.array([final_name_to_num[i] for i in test_index])
    name_test, num_test = test_dataset[:, 0], test_dataset[:, 1]
    train_dataset = np.array([final_name_to_num[i] for i in train_index])
    name_train, name_val, num_train, num_val = train_test_split(
        train_dataset[:, 0],
        train_dataset[:, 1],
        stratify=train_dataset[:, 1],
        test_size=params["val_split"],
        random_state=params["random_seed"]
    )
    return name_train, num_train, name_val, num_val, name_test, num_test

def generate_augmented_data(name_train, num_train, paths, fold_counter):
    name_train = np.array(name_train)
    num_train = np.array(num_train)
    orig_name_train, orig_num_train = [], []
    DA_name_train, DA_num_train = [], []
    for label in np.unique(num_train):
        idxs = np.where(num_train == label)[0]

        if int(label) in constants.ORIGINAL_DA_NUM_TO_LABEL.keys():
            n = len(idxs)
            n_replace = n // 2
            replace_idxs = idxs[:n_replace]
            keep_idxs = idxs[n_replace:]
            img_paths = [os.path.join(paths["images_dir"], name_train[idx]) for idx in replace_idxs]
            augmented_imgs = traditional_DA(img_paths, constants.AUGMENTATION_PARAMS)
            aug_dir = os.path.join(paths["augmented_images_dir"], str(label))
            os.makedirs(aug_dir, exist_ok=True)
            for i, aug_img in enumerate(augmented_imgs):
                aug_name = f"aug_{fold_counter}_{i}_{label}.jpg"
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

def pack_train_val_test(orig_name_train, orig_num_train, DA_name_train, DA_num_train, name_val, num_val, name_test, num_test):
    return {
        "train": {"name": orig_name_train, "num": orig_num_train},
        "DA_train": {"name": DA_name_train, "num": DA_num_train},
        "val": {"name": name_val, "num": num_val},
        "test": {"name": name_test, "num": num_test}
    }

def process_fold(fold_counter, final_name_to_num, train_index, test_index, num_to_label, params, paths):
    name_train, num_train, name_val, num_val, name_test, num_test = split_fold_data(
        final_name_to_num, train_index, test_index, params
    )

    if constants.DA_METHOD != None:
        orig_name_train, orig_num_train, DA_name_train, DA_num_train = generate_augmented_data(
            name_train, num_train, paths, fold_counter
        )
    else:
        orig_name_train = name_train
        orig_num_train = num_train
        DA_name_train = np.array([])
        DA_num_train = np.array([])

    train_val_test_pack = pack_train_val_test(
        orig_name_train, orig_num_train, DA_name_train, DA_num_train, name_val, num_val, name_test, num_test
    )

    move_img(train_val_test_pack, num_to_label, paths)
    plot_class_dist(final_name_to_num, num_train, num_val, num_test, num_to_label, fold_counter)
    
    y_true_pred_dic = yolo_classify(fold_counter, params)
    move_back_img(paths)
    return y_true_pred_dic

def cross_validate(final_name_to_num, num_to_label, params, paths):
    move_back_img(paths)
    skf = StratifiedKFold(n_splits=params["k_folds"])
    all_y_true_pred = {"true": [], "pred": []}
    time = str(datetime.datetime.today())
    # constants.set_YOLO_Result(f"{paths['YOLO_result_root']}/{time}")
    constants.set_YOLO_Result(f"{paths['YOLO_result_root']}/{constants.DA_METHOD}/seed_{params["random_seed"]}/option_{OPTION_num}")

    os.makedirs(constants.YOLO_RESULT, exist_ok=True)
    for fold_counter, (train_index, test_index) in enumerate(
        skf.split(final_name_to_num[:, 0], final_name_to_num[:, 1].astype(int)), 1):
        y_true_pred_dic = process_fold(
            fold_counter, final_name_to_num, train_index, test_index, num_to_label, params, paths
        )
        all_y_true_pred["true"].extend(y_true_pred_dic["true"])
        all_y_true_pred["pred"].extend(y_true_pred_dic["pred"])
    return all_y_true_pred

def evaluate_and_save(all_y_true_pred, num_to_label):
    acc = accuracy_score(all_y_true_pred["true"], all_y_true_pred["pred"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_y_true_pred["true"], all_y_true_pred["pred"], average='macro'
    )
    total_evaluation = {"accuracy": acc, "precison": precision, "recall": recall, "f1-score": f1}

    with open(f"{constants.YOLO_RESULT}/true_pred.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(all_y_true_pred.keys())
        writer.writerows(zip(*all_y_true_pred.values()))


    with open(f"{constants.YOLO_RESULT}/total_evaluation.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(total_evaluation.keys())
        writer.writerows([list(total_evaluation.values())])

    all_y_true_pred["true"] = np.array(all_y_true_pred["true"]).flatten()
    all_y_true_pred["pred"] = np.array(all_y_true_pred["pred"]).flatten()
    labels = list(num_to_label.values())
    plot_confusion_matrix(all_y_true_pred["true"], all_y_true_pred["pred"], labels, f"{constants.YOLO_RESULT}/Confusion Matrix.png")

def main():
    paths = get_paths()
    params = get_params()
    seed_list = [12,22,32,52,62,72,82,92,102] # 42
    number_of_NON_AI_DA_METHOD = len(constants.NON_AI_DA_PARAMS)
    for s in seed_list:
        params["random_seed"] = s
        for n in range(number_of_NON_AI_DA_METHOD):
            global OPTION_num
            OPTION_num = n
            constants.set_augmentation_params(constants.NON_AI_DA_PARAMS[OPTION_num])
            constants.set_DA_method("NON_AI_BASED")
            final_name_to_num, num_to_label = prepare_data(params, paths)

            all_y_true_pred = cross_validate(final_name_to_num, num_to_label, params, paths)
            evaluate_and_save(all_y_true_pred, num_to_label)

if __name__ == "__main__":
    main()