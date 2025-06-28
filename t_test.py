import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math

def plot_f1_score_dist(df):
    n_models = len(df)
    n_cols = 4
    n_rows = math.ceil(n_models / 2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2 * n_rows))
    axes = axes.reshape(-1, n_cols)

    for i, (model_name, row) in enumerate(df.iterrows()):
        row_idx = i // 2
        col_idx = (i % 2) * 2

        sns.histplot(row.values, kde=True, ax=axes[row_idx, col_idx], bins=5, stat="density")
        axes[row_idx, col_idx].set_title(f"{model_name} - Histogram")
        axes[row_idx, col_idx].set_xlabel("F1 Score")

        stats.probplot(row.values, dist="norm", plot=axes[row_idx, col_idx + 1])
        axes[row_idx, col_idx + 1].set_title(f"{model_name} - Q-Q Plot")

    for j in range(i + 1, n_rows * 2):
        row_idx = j // 2
        col_idx = (j % 2) * 2
        axes[row_idx, col_idx].axis("off")
        axes[row_idx, col_idx + 1].axis("off")

    plt.tight_layout()

    plt.savefig("t_test/f1-score_dist.png", dpi=300)


gan_based = "YOLO_result/GAN_BASED/2025-06-25 16:04:36.997979"
non_ai_root = "YOLO_result/NON_AI_BASED/seed_42/"
option_list = ["option_0", "option_1", "option_2", "option_3", "option_4", "option_5"]

non_ai_path_dic = {path: os.path.join(non_ai_root, path) for path in option_list}

f1_score_list = []

for path in non_ai_path_dic.values():
    row = {}
    for i in range(1, 6):
        file_path = os.path.join(path, f"evaluation_fold_{i}.csv")
        row[f"fold{i}"] = pd.read_csv(file_path)["f1-score"].values[0]
    f1_score_list.append(row)


gan_row = {}
for i in range(1, 6):
    file_path = os.path.join(gan_based, f"evaluation_fold_{i}.csv")
    gan_row[f"fold{i}"] = pd.read_csv(file_path)["f1-score"].values[0]
f1_score_list.append(gan_row)


df = pd.DataFrame(f1_score_list, index=option_list + ["GAN_BASED"])
print("f1 scores------------------------------------------------------------")
print(df)
print("---------------------------------------------------------------------\n\n")


# plot_f1_score_dist(df)

models = df.index.tolist()
n = len(models)

pvals = pd.DataFrame(index=models, columns=models, dtype=float)

for i in range(n):
    for j in range(n):
        if i == j:
            pvals.iloc[i, j] = None
        else:
            _, p_value = stats.ttest_rel(df.iloc[i], df.iloc[j])
            pvals.iloc[i, j] = p_value

print("Paired t-test p-values------------------------------------------------------------")
print(pvals)
print("----------------------------------------------------------------------------------")

significant_pairs = (pvals < 0.05)
print("Significant differences (p < 0.05):")
print(significant_pairs)
