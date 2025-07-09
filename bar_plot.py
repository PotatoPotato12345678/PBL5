import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths
gan_based = "YOLO_result/GAN_BASED/2025-06-25 16:04:36.997979"
non_ai_root = "YOLO_result/NON_AI_BASED/seed_42/"
option_list = ["option_0", "option_1", "option_2", "option_3", "option_4", "option_5"]
method_list = ["method 1", "method 2", "method 3", "method 4", "method 5", "method 6"]
non_da_balance = "YOLO_result/NON_DA_BALANCE/2025-07-03 18:28:44.687156"
non_da_imbalance = "YOLO_result/NON_DA_IMBALANCE/2025-07-03 18:20:32.650160"

# Mapping option_x to method names
non_ai_path_dic = {method_list[i]: os.path.join(non_ai_root, option_list[i]) for i in range(6)}

# Collect F1-scores
f1_score_list = []
methods = []
da_types = []

def getF1score(path):
    file_path = os.path.join(path, "total_evaluation.csv")
    return pd.read_csv(file_path)["f1-score"].values[0]

# Non-AI-Based methods
for method, path in non_ai_path_dic.items():
    f1_score_list.append(getF1score(path))
    methods.append(method)
    da_types.append("Non-AI-Based")

# GAN-Based method
f1_score_list.append(getF1score(gan_based))
methods.append("DCGAN + CGAN")
da_types.append("GAN-Based")

# No DA - Full set
f1_score_list.append(getF1score(non_da_balance))
methods.append("Full reference set")
da_types.append("No DA")

# No DA - Reduced set
f1_score_list.append(getF1score(non_da_imbalance))
methods.append("Reduced reference set")
da_types.append("No DA")


df = pd.DataFrame({
    "Methods": methods,
    "DA Type": da_types,
    "F1-score": f1_score_list
})

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=df,
    x="Methods",
    y="F1-score",
    hue="DA Type",
    palette="Set2"
)
plt.xticks(rotation=30, ha='right')
plt.ylim(0.80, 0.90)

for container in barplot.containers:
    barplot.bar_label(container, fmt='%.3f', padding=3)


plt.tight_layout()


plt.savefig("./result_img/f1_score_plot.png", dpi=300, bbox_inches='tight')