def get_params():
    return {
        "move_back_flag": True,
        "quick_demo_flag": False,
        "method_selection": "down_sampling",
        "down_sampling_size": 5,
        "stratified_size": 1/10,
        "k_folds": 5,
        "random_seed": 42,
        "val_split": 0.25,
        "epochs": 1,
        "imgsz": 640,
        "device": "cpu"
    }