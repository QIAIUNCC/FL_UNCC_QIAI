import os
import torch
from dotenv import load_dotenv
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
from fl_config import get_dataloaders, log_results
from utils.data_handler import get_datasets_classes
from utils.plot import plot_auc
from utils.utils import set_seed, get_img_transformation, get_hyperparameters
from models.resnet import ResNet
from models.simple_vit import ViT


def train_local_model(train_loader, val_loader, mode, monitor, name_suffix, classes, client_name,
                      kermany_test_loader, srinivasan_test_loader, oct500_test_loader, patience=10, batch_size=400,
                      model_architecture="resnet18", max_epochs=10, model=None):
    early_stopping = EarlyStopping(monitor=monitor, patience=patience, verbose=False, mode=mode)
    model_path = os.path.join(CHECKPOINT_PATH, f"{name_suffix}_{model_architecture}")
    param = get_hyperparameters(client_name, model_architecture)
    if "resnet" in model_architecture:
        model = ResNet(classes=classes,
                       lr=param["lr"],
                       weight_decay=param["wd"],
                       architecture=model_architecture,
                       optimizer="AdamW",
                       beta1=param["beta1"],
                       beta2=param["beta2"]
                       )
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
        model = ResNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:

        model = ViT(classes=classes,
                    beta1=param["beta1"],
                    beta2=param["beta2"],
                    optimizer="AdamW",
                    lr=param["lr"],
                    weight_decay=param["wd"],
                    )
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
    # config = {"train_batch_size": batch_size, "max_epochs": max_epochs, 'current_round': 10}
    # log_name = f'log/{name_suffix}_{model_architecture}_{batch_size}'
    # kermany_test_results = trainer.test(model, kermany_test_loader, verbose=False)
    # log_results(classes, kermany_test_results, client_name, model_architecture,
    #             config, log_suffix="Kermnay", approach="local")
    # srinivasan_test_results = trainer.test(model, srinivasan_test_loader, verbose=False)
    # log_results(classes, srinivasan_test_results, client_name, model_architecture,
    #             config, log_suffix="Srinivasan", approach="local")
    # oct500_test_results = trainer.test(model, oct500_test_loader, verbose=False)
    # log_results(classes, oct500_test_results, client_name, model_architecture,
    #             config, log_suffix="OCT-500", approach="local")

    # plot_auc(model=model,
    #          log_name=log_name + "_" + str(max_epochs),
    #          test_datasets={
    #              "Kermany": kermany_test_loader,
    #              "Srinivasan": srinivasan_test_loader,
    #              "OCT-500": oct500_test_loader,
    #          },
    #          title=client_name)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # set_seed(10)
    torch.set_float32_matmul_precision('medium')

    NUM_WORKERS = os.cpu_count() // 2
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    load_dotenv(dotenv_path="../data/.env")
    DATASET_PATH = os.getenv('DATASET_PATH')
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "../saved_models/local_model/"
    batch_size = 64
    kermany_classes, srinivasan_classes, oct500_classes = get_datasets_classes()
    kermany_train_loader, kermany_val_loader, \
        kermany_test_loader, classes = get_dataloaders(cid="0",
                                                       dataset_path=DATASET_PATH,
                                                       batch_size=batch_size,
                                                       kermany_classes=kermany_classes,
                                                       srinivasan_classes=srinivasan_classes,
                                                       oct500_classes=oct500_classes,
                                                       img_transforms=get_img_transformation(),
                                                       limit_kermany=False
                                                       )
    srinivasan_train_loader, srinivasan_val_loader, \
        srinivasan_test_loader, _ = get_dataloaders(cid="1",
                                                    dataset_path=DATASET_PATH,
                                                    batch_size=batch_size,
                                                    kermany_classes=kermany_classes,
                                                    srinivasan_classes=srinivasan_classes,
                                                    oct500_classes=oct500_classes,
                                                    img_transforms=get_img_transformation(),
                                                    limit_kermany=False
                                                    )
    oct500_train_loader, oct500_val_loader, \
        oct500_test_loader, _ = get_dataloaders(cid="2",
                                                dataset_path=DATASET_PATH,
                                                batch_size=batch_size,
                                                kermany_classes=kermany_classes,
                                                srinivasan_classes=srinivasan_classes,
                                                oct500_classes=oct500_classes,
                                                img_transforms=get_img_transformation(),
                                                limit_kermany=False
                                                )
    #
    mode = "max"
    monitor = "val_auc"
    max_epochs = 1
    model_architecture = "resnet18"
    # model_architecture = "ViT"
    count_0 = 0
    count_1 = 0

    # for batch in oct500_test_loader:
    #     labels = batch['label']
    #     count_0 += (labels == 0).sum().item()
    #     count_1 += (labels == 1).sum().item()
    #
    # print(f"Number of entries with label 0: {count_0}")
    # print(f"Number of entries with label 1: {count_1}")
    for i in range(0, 1):
        # train_local_model(kermany_train_loader, kermany_val_loader, mode=mode, monitor=monitor, client_name="Kermany",
        #                   name_suffix="kermany_local_model", classes=kermany_classes, batch_size=batch_size,
        #                   kermany_test_loader=kermany_test_loader, srinivasan_test_loader=srinivasan_test_loader,
        #                   oct500_test_loader=oct500_test_loader, patience=10, max_epochs=max_epochs,
        #                   model_architecture=model_architecture)

        # train_local_model(srinivasan_train_loader, srinivasan_val_loader, mode=mode, monitor=monitor,
        #                   client_name="Srinivasan",
        #                   name_suffix="srinivasan_local_model", classes=srinivasan_classes, batch_size=batch_size,
        #                   kermany_test_loader=kermany_test_loader, srinivasan_test_loader=srinivasan_test_loader,
        #                   oct500_test_loader=oct500_test_loader, patience=10, max_epochs=max_epochs,
        #                   model_architecture=model_architecture)

        train_local_model(oct500_train_loader, oct500_val_loader, mode=mode, monitor=monitor, client_name="OCT-500",
                          name_suffix="oct500_local_model", classes=oct500_classes, batch_size=batch_size,
                          kermany_test_loader=kermany_test_loader, srinivasan_test_loader=srinivasan_test_loader,
                          oct500_test_loader=oct500_test_loader, patience=10, max_epochs=max_epochs,
                          model_architecture=model_architecture)
