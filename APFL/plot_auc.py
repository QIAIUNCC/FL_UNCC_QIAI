import os
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from FedAP.resnet_APFL import ResNetAP
from fl_config import log_results
from hyperparameters import get_centralized_AdamW_ResNet_parameters, get_AP_config
from utils.data_handler import get_oct500_datasets, get_srinivasan_datasets, get_data_loaders, get_datasets_classes, \
    get_kermany_datasets
from utils.plot import plot_auc
from utils.utils import set_seed, get_img_transformation

if __name__ == "__main__":
    set_seed(10)
    NUM_WORKERS = 4
    kermany_classes, srinivasan_classes, oct500_classes = get_datasets_classes()
    batch_size = 64
    load_dotenv(dotenv_path="./../data/.env")
    DATASET_PATH = os.getenv('DATASET_PATH')
    _, _, kermany_dataset_test = get_kermany_datasets(
        DATASET_PATH + "/0/train",
        DATASET_PATH + "/0/test",
        kermany_classes,
        img_transformation=get_img_transformation(),
        val_split=0.05,
    )

    _, _, oct500_dataset_test_6mm = \
        get_oct500_datasets(DATASET_PATH + "/2/OCTA_6mm", oct500_classes, img_transformation=get_img_transformation())
    _, _, oct500_dataset_test_3mm = \
        get_oct500_datasets(DATASET_PATH + "/2/OCTA_3mm", oct500_classes, img_transformation=get_img_transformation())

    oct500_dataset_test = torch.utils.data.ConcatDataset([oct500_dataset_test_6mm, oct500_dataset_test_3mm])

    srinivasan_dataset_train, srinivasan_dataset_val, \
        srinivasan_dataset_test = get_srinivasan_datasets(DATASET_PATH + "/1/train",
                                                          DATASET_PATH + "/1/test",
                                                          srinivasan_classes,
                                                          img_transformation=
                                                          get_img_transformation())

    kermany_test_loader = DataLoader(kermany_dataset_test, batch_size=1, shuffle=False,
                                     drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    srinivasan_test_loader = DataLoader(srinivasan_dataset_test, batch_size=1, shuffle=False,
                                        drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    oct500_test_loader = DataLoader(oct500_dataset_test, batch_size=1, shuffle=False,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=False,
    )
    adam_config = get_centralized_AdamW_ResNet_parameters()
    architecture = "resnet18"
    ap_config = get_AP_config()
    model = ResNetAP(classes=kermany_classes,
                     lr=adam_config["lr"],
                     weight_decay=adam_config["wd"],
                     optimizer="AdamW",
                     beta1=adam_config["beta1"],
                     beta2=adam_config["beta2"],
                     alpha=ap_config["alpha"],
                     architecture=architecture
                     )
    state_dict = torch.load('./ResNet/model_round_10.pth')
    model.model.load_state_dict(state_dict, strict=True)

    kermany_test_results = trainer.test(model, kermany_test_loader, verbose=False)
    srinivasan_test_results = trainer.test(model, srinivasan_test_loader, verbose=False)
    oct500_test_results = trainer.test(model, oct500_test_loader, verbose=False)
    log_name = f'log/{"FL_AP"}_{architecture}_{batch_size}'

    plot_auc(model=model,
             log_name=log_name + "_" + str(5),
             test_datasets={"Kermany": kermany_test_loader,
                            "Srinivasan": srinivasan_test_loader,
                            "OCT-500": oct500_test_loader},
             title="Centralized " + architecture)
