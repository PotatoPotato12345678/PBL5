import os
import sys
import numpy as np
import random
import datetime
import csv
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from constants import get_paths, get_params
from augmentation import generate_augmented_data
import constants
from constants import get_paths, get_params
from data_utils import move_back_img, create_num_to_label, create_name_to_num, move_img,plot_class_dist,pack_train_val_test, plot_confusion_matrix, remove_half_data
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

def process_fold(fold_counter, final_name_to_num, train_index, test_index, num_to_label, params, paths):
    name_train, num_train, name_val, num_val, name_test, num_test = split_fold_data(
        final_name_to_num, train_index, test_index, params
    )

    if constants.DA_METHOD == "GAN_BASED" or constants.DA_METHOD == "NON_AI_BASED":
        orig_name_train, orig_num_train, DA_name_train, DA_num_train = generate_augmented_data(
            name_train, num_train, paths, fold_counter
        )
    elif constants.DA_METHOD == "NON_DA_FULL":
        orig_name_train = name_train
        orig_num_train = num_train
        DA_name_train = np.array([])
        DA_num_train = np.array([])
    elif constants.DA_METHOD == "NON_DA_IMBALANCE":
        orig_name_train, orig_num_train, DA_name_train, DA_num_train = remove_half_data(num_train, name_train)

    elif constants.DA_METHOD == "NON_DA_FULL":
        orig_name_train = name_train
        orig_num_train = num_train
        DA_name_train = np.array([])
        DA_num_train = np.array([])
    else:
        print("DA method is not correct")
        sys.exit(1)


    train_val_test_pack = pack_train_val_test(
        orig_name_train, orig_num_train, DA_name_train, DA_num_train, name_val, num_val, name_test, num_test
    )

    move_img(train_val_test_pack, num_to_label, paths)

    plot_class_dist(final_name_to_num, np.concatenate((orig_num_train, DA_num_train), axis=0), num_val, num_test, num_to_label, fold_counter)

    y_true_pred_dic = yolo_classify(fold_counter, params)
    move_back_img(paths)
    return y_true_pred_dic

def cross_validate(final_name_to_num, num_to_label, params, paths):
    move_back_img(paths)
    skf = StratifiedKFold(n_splits=params["k_folds"])
    all_y_true_pred = {"true": [], "pred": []}
    time = str(datetime.datetime.today())
    # constants.set_YOLO_Result(f"{paths['YOLO_result_root']}/{time}")
    constants.set_YOLO_Result(f"{paths['YOLO_result_root']}/{constants.DA_METHOD}/{time}")

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
    params["random_seed"] = 4

    DA_method = "NON_DA_FULL" # NON_DA_IMBALANCE, NON_DA_FULL, NON_AI_BASED, GAN_BASED

    if DA_method == "GAN_BASED":
        constants.set_DA_method("GAN_BASED")
    elif DA_method == "NON_AI_BASED":
        constants.set_DA_method("NON_AI_BASED")
        constants.set_augmentation_params(constants.NON_AI_DA_PARAMS[1])
    elif DA_method == "NON_DA_FULL":
        constants.set_DA_method("NON_DA_FULL")
    elif DA_method == "NON_DA_IMBALANCE":
        constants.set_DA_method("NON_DA_IMBALANCE")    
    else:
        print("Please put the correct DA methods.")
        sys.exit(1)

    final_name_to_num, num_to_label = prepare_data(params, paths)

    all_y_true_pred = cross_validate(final_name_to_num, num_to_label, params, paths)
    evaluate_and_save(all_y_true_pred, num_to_label)

if __name__ == "__main__":
    main()
