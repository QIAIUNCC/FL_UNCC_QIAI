import os
import shutil

import pytorch_lightning as pl
from dotenv import load_dotenv
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from skopt.space import Real, Categorical
from models.simple_vit import ViT
from utils.data_handler import get_datasets_classes, get_data_loaders, get_srinivasan_datasets, get_oct500_datasets, \
    get_kermany_datasets
from utils.utils import set_seed, get_img_transformation
import torch
from sklearn.model_selection import ParameterSampler
import numpy as np

set_seed(10)

param_dist = {
    'learning_rate': list(np.logspace(-6, -3, 100)),  # Log-uniform distribution
    'weight_decay': list(np.logspace(-6, -3, 100)),  # Uniform distribution
    'beta1': list(np.linspace(0.8, 0.9, 10)),  # Uniform distribution
    'beta2': list(np.linspace(0.9, 0.99, 10)),  # Uniform distribution
}
n_iter_search = 100
param_list = list(ParameterSampler(param_dist, n_iter=n_iter_search))
batch_size = 64
best_score = -np.inf

load_dotenv(dotenv_path="../data/.env")
DATASET_PATH = os.getenv('DATASET_PATH')
kermany_classes, srinivasan_classes, oct500_classes = get_datasets_classes()
optimizer = "AdamW"
log_dir = f"./global/ViT"
for params in param_list:
    # batch_size = params["batch_size"]
    # dampening = params["dampening"]
    # momentum = params["momentum"]
    # mu = params["mu"]

    beta1 = params["beta1"]
    beta2 = params["beta2"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]

    # Set up model with the current hyperparameters
    model = ViT(classes=kermany_classes,
                lr=learning_rate,
                weight_decay=weight_decay,
                # architecture="resnet18",
                optimizer=optimizer,
                # momentum=momentum,
                # dampening=dampening,
                beta1=beta1,
                beta2=beta2,
                )

    kermany_dataset_train, kermany_dataset_val, _ = get_kermany_datasets(
        DATASET_PATH + "/0/train",
        DATASET_PATH + "/0/test", kermany_classes, img_transformation=get_img_transformation(), val_split=0.05,
    )

    oct500_dataset_train_6mm, oct500_dataset_val_6mm, _ = \
        get_oct500_datasets(DATASET_PATH + "/2/OCTA_6mm", oct500_classes, img_transformation=get_img_transformation())
    oct500_dataset_train_3mm, oct500_dataset_val_3mm, _ = \
        get_oct500_datasets(DATASET_PATH + "/2/OCTA_3mm", oct500_classes, img_transformation=get_img_transformation())

    oct500_dataset_train = torch.utils.data.ConcatDataset([oct500_dataset_train_6mm, oct500_dataset_train_3mm])
    oct500_dataset_val = torch.utils.data.ConcatDataset([oct500_dataset_val_6mm, oct500_dataset_val_3mm])

    srinivasan_dataset_train, srinivasan_dataset_val, _ = get_srinivasan_datasets(DATASET_PATH + "/1/train",
                                                                                  DATASET_PATH + "/1/test",
                                                                                  srinivasan_classes,
                                                                                  img_transformation=
                                                                                  get_img_transformation())
    train_dataset = torch.utils.data.ConcatDataset(
        [kermany_dataset_train, srinivasan_dataset_train, oct500_dataset_train])
    val_dataset = torch.utils.data.ConcatDataset([kermany_dataset_val, srinivasan_dataset_val, oct500_dataset_val])
    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset, int(batch_size))

    early_stopping = EarlyStopping(monitor="val_auc", patience=10, verbose=False, mode="max")
    mc = ModelCheckpoint(dirpath=log_dir, save_weights_only=True, mode="max",
                         monitor="val_auc", save_top_k=1)
    trainer = pl.Trainer(accelerator="gpu",
                         devices=[1],
                         max_epochs=10,
                         logger=False,
                         callbacks=[early_stopping, mc]
                         )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, )
    model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    results = trainer.test(model, val_loader, verbose=False)

    if results[0]["test_auc"] > best_score:
        best_score = results[0]["test_auc"]
        best_params = params
    shutil.rmtree(log_dir)

print(optimizer, "Centralized ViT", "random search")
print("Best parameters: ", best_params)
print("Best score: ", best_score)
