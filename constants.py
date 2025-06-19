YOLOV8_IMAGE_SIZE = 640
YOLOV8_BATCH_SIZE = 16
YOLOV8_EPOCHS = 50

# TRAIN_RATIO = 0.6
# VAL_RATIO = 0.2
# TEST_RATIO = 0.2 # 5-fold

TRAIN_SIZE_PER_CLASS = 240

ORIGINAL_DA_NUM_TO_LABEL = {
    23: "corn borer",
    52: "blister beetle"
}

N_CLASS = 4

# Constant for CGAN
GAN_PARAMS = {
    "TRAIN_SIZE": TRAIN_SIZE_PER_CLASS * 2,
    "BATCH_SIZE": 64,
    "IMG_SIZE": 128,
    "EPOCH_COUNT": 100,
    "NOISE_DIM": 100,
    "N_CLASS": 2,
    "L_RATE_G": 0.0001,
    "L_RATE_D": 0.0004,
    "L_RATE_C": 0.0004,
    "B1": 0.5,
    "B2": 0.999,
    "R": 0.2,
    "WEIGHT_DECAY": 1e-4
}

AUGMENTATION_PARAMS = {
    "rotation_range": None,
    "brightness_range":None,
    "contrast_range": None,
    "shear_range": None,
}

DA_METHOD = None # NON_AI_BASED, GAN_BASED


def set_DA_method(method):
    global DA_METHOD
    DA_METHOD = method

def set_augmentation_params(prms):
    global AUGMENTATION_PARAMS
    AUGMENTATION_PARAMS["rotation_range"] = prms["rotation_range"]
    AUGMENTATION_PARAMS["brightness_range"] = prms["brightness_range"]
    AUGMENTATION_PARAMS["contrast_range"] = prms["contrast_range"]
    AUGMENTATION_PARAMS["shear_range"] = prms["shear_range"]


NON_AI_DA_PARAMS = [
{ "rotation_range": 150,
    "brightness_range":1.9,
    "contrast_range": 5,
    "shear_range": 0,},
{ "rotation_range": 150,
    "brightness_range":2,
    "contrast_range": 5,
    "shear_range": 0,},
{ "rotation_range": 140,
    "brightness_range":2,
    "contrast_range": 5,
    "shear_range": 0,},
{ "rotation_range": 140,
    "brightness_range":1.9,
    "contrast_range": 5,
    "shear_range": 0,},
{ "rotation_range": 140,
    "brightness_range":2,
    "contrast_range": 4.5,
    "shear_range": 0,},
{ "rotation_range": 150,
    "brightness_range":2,
    "contrast_range": 5,
    "shear_range": 40,}]

def get_paths():
    return {
        "dataset_root": "IP102/dataset",
        "images_dir": "IP102/ip102_v1.1/images/",
        "augmented_images_dir": "IP102/ip102_v1.1/augmented_images/",
        "num_label_file": "IP102/ip102_v1.1/num_label_reduced.txt",
        "name_num_table": "IP102/ip102_v1.1/name_num_table.txt",
        "YOLO_result_root": "YOLO_result",
        "CGAN_result_root": "CGAN_result"
    }

YOLO_RESULT = None

def set_YOLO_Result(dir):
    global YOLO_RESULT
    YOLO_RESULT = dir


def get_params():
    return {
        "move_back_flag": True,
        "quick_demo_flag": True,
        "method_selection": "down_sampling",
        "down_sampling_size": TRAIN_SIZE_PER_CLASS, # per class
        "stratified_size": 1/10,
        "k_folds": 5,
        "random_seed": None, # 42
        "val_split": 0.25, # 6-2-2
        "epochs": 100,
        "batch": 128,
        "imgsz": 128,
        "device": "0"
    }