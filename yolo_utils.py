import os
import json
import csv
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import constants


def yolo_classify(fold_count, params):
    with open(f"{constants.YOLO_RESULT}/YOLO_BASIC_PARAMS.json", "w") as f:
        json.dump(params, f, indent=4) 

    if constants.DA_METHOD == "NON_AI_BASED":
        with open(f"{constants.YOLO_RESULT}/NON_AI_BASED_paramter.json", "w") as f:
            json.dump(constants.AUGMENTATION_PARAMS, f, indent=4)
    
    if constants.DA_METHOD == "GAN_BASED":
        with open(f"{constants.YOLO_RESULT}/GAN_BASED_paramter.json", "w") as f:
            json.dump(constants.GAN_PARAMS, f, indent=4)

    model = YOLO("yolov8n-cls.pt")

    train_results = model.train(
        data=os.path.join(os.getcwd(), "IP102/dataset"),
        # data=os.path.join(os.getcwd(), "data.yaml"),
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        device=params["device"],
        batch=params["batch"],
        seed=params["random_seed"]
    )

    with open(f"{constants.YOLO_RESULT}/yolov8_config.json", "w") as f:
        json.dump(model.overrides, f, indent=4)
    
    metrics = model.val()
    test_dir = os.path.join(os.getcwd(), "IP102/dataset/test")
    y_true, y_pred = [], []
    class_names = model.names
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            results = model.predict(image_path, verbose=False)
            pred_idx = results[0].probs.top1
            pred_name = class_names[pred_idx]
            y_true.append(class_name)
            y_pred.append(pred_name)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    fold_evaluation = {"accuracy": acc, "precison": precision, "recall": recall, "f1-score": f1}
    with open(f"{constants.YOLO_RESULT}/evaluation_fold_{fold_count}.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fold_evaluation.keys())
        writer.writerows([fold_evaluation.values()])
    return {"true": y_true, "pred": y_pred}