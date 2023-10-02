import os
import shutil
import pytorch_lightning as pl
from dotenv import load_dotenv
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import ParameterSampler
import numpy as np
from fl_config import get_dataloaders
from utils.data_handler import get_datasets_classes
from utils.utils import set_seed, get_img_transformation
from models.resnet import ResNet

set_seed(10)
# Define the hyperparameter configuration space
param_dist = {
    'learning_rate': list(np.logspace(-6, -3, 100)),  # Log-uniform distribution
    'weight_decay': list(np.logspace(-6, -3, 100)),  # Uniform distribution
    'beta1': list(np.linspace(0.8, 0.9, 10)),  # Uniform distribution
    'beta2': list(np.linspace(0.9, 0.99, 10)),  # Uniform distribution
}

n_iter_search = 100
param_list = list(ParameterSampler(param_dist, n_iter=n_iter_search))
best_score = -np.inf

load_dotenv(dotenv_path="../data/.env")
DATASET_PATH = os.getenv('DATASET_PATH')
kermany_classes, srinivasan_classes, oct500_classes = get_datasets_classes()
cid = "0"
log_dir = f"./localModel/{cid}/resent18"
optimizer = "AdamW"
batch_size = 64

for params in param_list:
    learning_rate = params["learning_rate"]
    beta1 = params["beta1"]
    weight_decay = params["weight_decay"]
    beta2 = params["beta2"]

    # Set up model with the current hyperparameters
    model = ResNet(classes=kermany_classes,
                   lr=learning_rate,
                   weight_decay=weight_decay,
                   architecture="resnet18",
                   optimizer=optimizer,
                   beta1=beta1,
                   beta2=beta2,
                   )

    train_loader, val_loader, _, classes = get_dataloaders(cid=cid,
                                                           dataset_path=DATASET_PATH,
                                                           batch_size=batch_size,
                                                           kermany_classes=kermany_classes,
                                                           srinivasan_classes=srinivasan_classes,
                                                           oct500_classes=oct500_classes,
                                                           img_transforms=get_img_transformation(),
                                                           limit_kermany=False
                                                           )
    early_stopping = EarlyStopping(monitor="val_auc", patience=10, verbose=False, mode="max")
    mc = ModelCheckpoint(dirpath=log_dir, save_weights_only=True, mode="max",
                         monitor="val_auc", save_top_k=1)

    trainer = pl.Trainer(default_root_dir=f"./localModel/{cid}",
                         accelerator="gpu",
                         devices=[0],
                         max_epochs=10,
                         logger=False,
                         callbacks=[early_stopping, mc]
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model = ResNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    results = trainer.test(model, val_loader, verbose=False)

    if results[0]["test_auc"] > best_score:
        best_score = results[0]["test_auc"]
        best_params = params
    shutil.rmtree(log_dir)

print(optimizer, "Cid", cid, "random search")
print("Best parameters: ", best_params)
print("Best score: ", best_score)
