import os
import numpy as np
import random
import datetime
import csv
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config.paths import get_paths
from config.params import get_params
from data_utils import move_back_img, create_num_to_label, create_name_to_num, move_img
from plot_utils import plot_class_dist, plot_confusion_matrix
from yolo_utils import yolo_classify

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
            min_size = min(len(names) for names in grouped.values())
            accept_min = int(np.floor(min_size / (3 * params["k_folds"])))
            if params["down_sampling_size"] > accept_min:
                params["down_sampling_size"] = accept_min
            random.seed(params["random_seed"])
            reduced_dataset_name = []
            reduced_dataset_num = []
            for num, names in grouped.items():
                reduced_dataset_name.extend(random.sample(names, params["down_sampling_size"] * 3 * params["k_folds"]))
                reduced_dataset_num.extend([num] * params["down_sampling_size"] * 3 * params["k_folds"])
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

def cross_validate(final_name_to_num, num_to_label, params, paths):
    move_back_img(paths)
    skf = StratifiedKFold(n_splits=params["k_folds"])
    all_y_true_pred = {"true": [], "pred": []}
    time = str(datetime.datetime.today())
    save_dir = f"{paths['result_root']}/{time}"
    os.makedirs(save_dir, exist_ok=True)
    for fold_counter, (train_index, test_index) in enumerate(skf.split(final_name_to_num[:, 0], final_name_to_num[:, 1].astype(int)), 1):
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
        train_val_test_pack = {
            "train": {"name": name_train, "num": num_train},
            "val": {"name": name_val, "num": num_val},
            "test": {"name": name_test, "num": num_test}
        }
        move_img(train_val_test_pack, num_to_label, paths)
        plot_class_dist(final_name_to_num, num_train, num_val, num_test, num_to_label, fold_counter, save_dir)
        y_true_pred_dic = yolo_classify(fold_counter, save_dir)
        all_y_true_pred["true"].extend(y_true_pred_dic["true"])
        all_y_true_pred["pred"].extend(y_true_pred_dic["pred"])
        move_back_img(paths)
    return all_y_true_pred, save_dir

def evaluate_and_save(all_y_true_pred, num_to_label, save_dir):
    acc = accuracy_score(all_y_true_pred["true"], all_y_true_pred["pred"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_y_true_pred["true"], all_y_true_pred["pred"], average='macro'
    )
    total_evaluation = {"accuracy": acc, "precison": precision, "recall": recall, "f1-score": f1}
    keys = all_y_true_pred.keys()
    rows = zip(*all_y_true_pred.values())
    with open(f"{save_dir}/true_pred.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)
    keys = total_evaluation.keys()
    rows = list(total_evaluation.values())
    with open(f"{save_dir}/total_evaluation.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows([rows])
    all_y_true_pred["true"] = np.array(all_y_true_pred["true"]).flatten()
    all_y_true_pred["pred"] = np.array(all_y_true_pred["pred"]).flatten()
    labels = list(num_to_label.values())
    plot_confusion_matrix(all_y_true_pred["true"], all_y_true_pred["pred"], labels, f"{save_dir}/Confusion Matrix.png")

def main():
    paths = get_paths()
    params = get_params()
    final_name_to_num, num_to_label = prepare_data(params, paths)
    all_y_true_pred, save_dir = cross_validate(final_name_to_num, num_to_label, params, paths)
    evaluate_and_save(all_y_true_pred, num_to_label, save_dir)

if __name__ == "__main__":
    main()