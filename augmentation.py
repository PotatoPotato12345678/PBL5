import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def traditional_DA(img_paths, prms):
    augmented_images = []
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')

        transform = transforms.Compose([
            transforms.RandomRotation(degrees=prms['rotation_range']),
            transforms.ColorJitter(
                brightness=prms['brightness_range'],
                contrast=prms['contrast_range']
            ),
            transforms.RandomAffine(
                degrees=0,
                shear=prms['shear_range']
            ),
            transforms.ToTensor()
        ])

        torch.manual_seed(0)
        transformed_img = transform(img)

        np_img = transformed_img.permute(1, 2, 0).numpy()
        np_img = np.clip(np_img, 0, 1)
        augmented_images.append(np_img)
    return augmented_images