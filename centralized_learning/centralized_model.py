import os
import torch
from dotenv import load_dotenv
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from hyperparameters import get_centralized_AdamW_ResNet_parameters, get_centralized_AdamW_ViT_parameters
from models.resnet import ResNet
from models.simple_vit import ViT
from utils.data_handler import get_oct500_datasets, get_srinivasan_datasets, get_datasets_classes, get_kermany_datasets, \
    get_data_loaders
from utils.utils import get_img_transformation


def train_model(architecture):
    if "resnet" in architecture.lower():
        param = get_centralized_AdamW_ResNet_parameters()
        model = ResNet(classes=kermany_classes,
                       lr=param["lr"],
                       weight_decay=param["wd"],
                       optimizer="AdamW",
                       beta1=param["beta1"],
                       beta2=param["beta2"],
                       architecture=architecture,
                       )
        early_stopping = EarlyStopping(monitor=monitor, patience=10, verbose=False, mode=mode)
        trainer = pl.Trainer(default_root_dir=model_path,
                             accelerator="gpu",
                             devices=[1],
                             max_epochs=max_epochs,
                             logger=False,
                             callbacks=[
                                 early_stopping,
                                 ModelCheckpoint(dirpath=model_path, save_weights_only=True, mode=mode,
                                                 monitor=monitor, save_top_k=1),
                             ])
        trainer.fit(model, train_loader, val_loader)

        model = ResNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                            architecture=model_architecture)
    else:
        param = get_centralized_AdamW_ViT_parameters()
        model = ViT(classes=kermany_classes,
                    lr=param["lr"],
                    weight_decay=param["wd"],
                    optimizer="AdamW",
                    beta1=param["beta1"],
                    beta2=param["beta2"]
                    )
        early_stopping = EarlyStopping(monitor=monitor, patience=10, verbose=False, mode=mode)
        trainer = pl.Trainer(default_root_dir=model_path,
                             accelerator="gpu",
                             devices=[0],
                             max_epochs=max_epochs,
                             logger=False,
                             callbacks=[
                                 early_stopping,
                                 ModelCheckpoint(dirpath=model_path, save_weights_only=True, mode=mode,
                                                 monitor=monitor, save_top_k=1),
                             ])
        trainer.fit(model, train_loader, val_loader)
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        #
    config = {"train_batch_size": batch_size, "max_epochs": max_epochs, 'current_round': 10}


if __name__ == "__main__":
    # set_seed(10)
    torch.set_float32_matmul_precision('medium')
    NUM_WORKERS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())]) * 4
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    load_dotenv(dotenv_path="../data/.env")
    DATASET_PATH = os.getenv('DATASET_PATH')
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "../saved_models/global_model/"
    kermany_classes, srinivasan_classes, oct500_classes = get_datasets_classes()

    batch_size = 64
    kermany_dataset_train, kermany_dataset_val, kermany_dataset_test = get_kermany_datasets(
        DATASET_PATH + "/0/train",
        DATASET_PATH + "/0/test",
        kermany_classes,
        img_transformation=get_img_transformation(),
        val_split=0.05,
    )

    oct500_dataset_train_6mm, oct500_dataset_val_6mm, oct500_dataset_test_6mm = \
        get_oct500_datasets(DATASET_PATH + "/2/OCTA_6mm", oct500_classes, img_transformation=get_img_transformation())
    oct500_dataset_train_3mm, oct500_dataset_val_3mm, oct500_dataset_test_3mm = \
        get_oct500_datasets(DATASET_PATH + "/2/OCTA_3mm", oct500_classes, img_transformation=get_img_transformation())

    oct500_dataset_train = torch.utils.data.ConcatDataset([oct500_dataset_train_6mm, oct500_dataset_train_3mm])
    oct500_dataset_val = torch.utils.data.ConcatDataset([oct500_dataset_val_6mm, oct500_dataset_val_3mm])
    oct500_dataset_test = torch.utils.data.ConcatDataset([oct500_dataset_test_6mm, oct500_dataset_test_3mm])

    srinivasan_dataset_train, srinivasan_dataset_val, \
        srinivasan_dataset_test = get_srinivasan_datasets(DATASET_PATH + "/1/train",
                                                          DATASET_PATH + "/1/test",
                                                          srinivasan_classes,
                                                          img_transformation=
                                                          get_img_transformation())
    train_dataset = torch.utils.data.ConcatDataset(
        [kermany_dataset_train, srinivasan_dataset_train, oct500_dataset_train])
    val_dataset = torch.utils.data.ConcatDataset([kermany_dataset_val, srinivasan_dataset_val, oct500_dataset_val])
    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset, int(batch_size))

    kermany_test_loader = DataLoader(kermany_dataset_test, batch_size=1, shuffle=False,
                                     drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    srinivasan_test_loader = DataLoader(srinivasan_dataset_test, batch_size=1, shuffle=False,
                                        drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    oct500_test_loader = DataLoader(oct500_dataset_test, batch_size=1, shuffle=False,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    #
    model_architecture = "ViT"
    # model_architecture = "ResNet18"
    name_suffix = "centralized_model"
    model_path = os.path.join(CHECKPOINT_PATH, f"{name_suffix}_{model_architecture}")
    mode = "max"
    monitor = "val_auc"
    max_epochs = 1

    for i in range(1, 2):
        train_model(architecture=model_architecture)
