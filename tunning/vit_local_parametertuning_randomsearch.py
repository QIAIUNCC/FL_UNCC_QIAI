import os
import shutil
import pytorch_lightning as pl
from dotenv import load_dotenv
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from skopt.space import Real, Categorical
from fl_config import get_dataloaders
from models.simple_vit import ViT
from utils.data_handler import get_datasets_classes
from utils.utils import set_seed, get_img_transformation
from sklearn.model_selection import ParameterSampler
import numpy as np

if __name__ == "__main__":
    # Define the dimensions of the hyperparameter search space
    space = [Real(1e-6, 1e-3, "log-uniform", name='learning_rate'),
             Real(1e-6, 1e-3, 'log-uniform', name='weight_decay'),
             Real(0.8, 0.999, name='beta1'),
             Real(0.9, 0.9999, name='beta2'),
             Categorical([32, 64, 128, 256], name="batch_size")
             ]

    set_seed(10)
    # Define the hyperparameter configuration space
    param_dist = {
        'learning_rate': list(np.logspace(-6, -3, 100)),  # Log-uniform distribution
        'weight_decay': list(np.logspace(-6, -3, 100)),  # Uniform distribution
        'beta1': list(np.linspace(0.8, 0.9, 10)),  # Uniform distribution
        'beta2': list(np.linspace(0.9, 0.99, 10)),  # Uniform distribution
        # 'batch_size': [32, 64, 128]  # Categorical distribution
    }

    n_iter_search = 100
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter_search))

    best_score = -np.inf

    load_dotenv(dotenv_path="../data/.env")
    DATASET_PATH = os.getenv('DATASET_PATH')
    kermany_classes, srinivasan_classes, oct500_classes = get_datasets_classes()
    cid = "0"
    optimizer = "AdamW"
    batch_size = 64

    log_dir = f"/home/sgholami/Desktop/projects/federated-learning-flower/tunning/localModel/{cid}/vit"
    for params in param_list:

        learning_rate = params["learning_rate"]
        beta1 = params["beta1"]
        weight_decay = params["weight_decay"]
        beta2 = params["beta2"]
        # batch_size = int(params["batch_size"])

        # Set up model with the current hyperparameters
        model = ViT(classes=kermany_classes,
                    lr=learning_rate,
                    weight_decay=weight_decay,
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
        trainer = pl.Trainer(accelerator="gpu",
                             devices=[1],
                             max_epochs=10,
                             logger=False,
                             callbacks=[mc, early_stopping]
                             )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, )
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        results = trainer.test(model, val_loader, verbose=False)

        if results[0]["test_auc"] > best_score:
            best_score = results[0]["test_auc"]
            best_params = params

        shutil.rmtree(log_dir)

    print(optimizer, "Cid", cid, "random search")
    print("Best parameters: ", best_params)
    print("Best score: ", best_score)
