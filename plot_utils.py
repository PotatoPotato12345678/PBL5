import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_class_dist(final_name_to_num, num_train, num_val, num_test, num_to_label, fold_count, save_dir):
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
        ax[x][y].set_xticklabels(ax[x][y].get_xticklabels(), fontsize=5)
        ax[x][y].set_yticklabels(ax[x][y].get_yticklabels(), fontsize=20)
    fig.tight_layout(pad=3.0)
    fig.savefig(f'{save_dir}/class_dist_fold_{fold_count}.png')
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