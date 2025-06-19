import numpy as np
from constants import ORIGINAL_DA_NUM_TO_LABEL, DA_LABEL_TO_NUM, SRC_DIRECTORY

def assign_augment_img(name_to_num):
    DA_path_to_num = []
    for name, num in zip(name_to_num[:, 0], name_to_num[:, 1]):
        num = int(num)
        if num in np.array(list(ORIGINAL_DA_NUM_TO_LABEL.keys())):
            DA_path_to_num.append([SRC_DIRECTORY + name, DA_LABEL_TO_NUM[ORIGINAL_DA_NUM_TO_LABEL[num]]])
    return np.array(DA_path_to_num)