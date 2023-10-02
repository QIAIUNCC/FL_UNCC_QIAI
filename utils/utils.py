import collections
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from flwr.common import Metrics
from torchvision.transforms import transforms, InterpolationMode

from hyperparameters import get_oct500_AdamW_ResNet_parameters, get_srinivasan_AdamW_ResNet_parameters, \
    get_kermany_AdamW_ResNet_parameters, get_oct500_AdamW_ViT_parameters, get_kermany_AdamW_ViT_parameters, \
    get_srinivasan_AdamW_ViT_parameters, get_centralized_AdamW_ResNet_parameters, get_centralized_AdamW_ViT_parameters


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print("============Server Aggregation=============")
    print(metrics)
    counter = collections.Counter()
    total_examples = sum([example[0] for example in metrics])
    # aggregate the metrics
    for m in metrics:
        counter.update({key: m[0] * value / total_examples for key, value in m[1].items()})
    result = dict(counter)
    # Aggregate and return custom metric (weighted average)
    return result


def get_img_transformation():
    img_transforms = transforms.Compose([
        transforms.Resize((128, 128), InterpolationMode.BICUBIC),
        transforms.RandomApply(torch.nn.ModuleList([transforms.ElasticTransform(alpha=(28.0, 30.0),
                                                                                sigma=(3.5, 4.0))]), p=0.3),
        transforms.RandomAffine(degrees=4.6, scale=(0.98, 1.02), translate=(0.03, 0.03)),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return img_transforms


def get_hyperparameters(client_name, model_architecture):
    if "resnet" in model_architecture:
        config = get_oct500_AdamW_ResNet_parameters()
        if client_name == "Kermany":
            config = get_kermany_AdamW_ResNet_parameters()
        elif client_name == "Srinivasan":
            config = get_srinivasan_AdamW_ResNet_parameters()
    else:
        config = get_oct500_AdamW_ViT_parameters()
        if client_name == "Kermany":
            config = get_kermany_AdamW_ViT_parameters()
        elif client_name == "Srinivasan":
            config = get_srinivasan_AdamW_ViT_parameters()

    return config
def set_seed(seed=10):
    pl.seed_everything(seed)
    np.random.seed(seed=seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.set_float32_matmul_precision('medium')
