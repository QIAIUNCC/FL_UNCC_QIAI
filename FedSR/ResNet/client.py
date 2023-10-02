from time import sleep

import torch
from dotenv import load_dotenv
from FedSR.resnet_FedSR import ResNetSR
from fl_client import main, FlowerClient
import os

from fl_config import get_dataloaders
from hyperparameters import get_SR_config
from utils.data_handler import get_datasets_classes
from utils.utils import get_img_transformation, set_seed, get_hyperparameters

load_dotenv(dotenv_path="../../data/.env")
DATASET_PATH = os.getenv('DATASET_PATH')
sr_config = get_SR_config()
architecture = "resnet18"
client_name = str(os.getenv('CLIENT_NAME'))
adam_config = get_hyperparameters(model_architecture=architecture, client_name=client_name)
kermany_classes, srinivasan_classes, oct500_classes = get_datasets_classes()


def client_fn_ResNetSR(cid: str) -> FlowerClient:
    """Creates a FlowerClient instance on demand
    Create a Flower client representing a single organization
    """
    set_seed(10)
    cid = "0"
    if client_name == "Srinivasan":
        cid = "1"
    elif client_name == "OCT-500":
        cid = "2"
    train_loader, val_loader, test_loader, classes = get_dataloaders(cid=cid,
                                                                     dataset_path=DATASET_PATH,
                                                                     batch_size=sr_config["batch_size"],
                                                                     kermany_classes=kermany_classes,
                                                                     srinivasan_classes=srinivasan_classes,
                                                                     oct500_classes=oct500_classes,
                                                                     img_transforms=get_img_transformation(),
                                                                     limit_kermany=False
                                                                     )
    train_loader, val_loader, test_loader, classes = get_dataloaders(cid=cid,
                                                                     dataset_path=DATASET_PATH,
                                                                     batch_size=sr_config["batch_size"],
                                                                     kermany_classes=kermany_classes,
                                                                     srinivasan_classes=srinivasan_classes,
                                                                     oct500_classes=oct500_classes,
                                                                     img_transforms=get_img_transformation(),
                                                                     limit_kermany=False
                                                                     )
    return FlowerClient(model, train_loader, val_loader, test_loader,
                        client_name=client_name, architecture=architecture)


if __name__ == "__main__":
    set_seed(10)
    server_ip = os.getenv('SERVER_IP')
    for i in range(1, 11):
        model = ResNetSR(classes=kermany_classes,
                         lr=adam_config["lr"],
                         weight_decay=adam_config["wd"],
                         z_dim=sr_config["z_dim"],
                         num_samples=sr_config["num_samples"],
                         CMI_coeff=sr_config["CMI_coeff"],
                         L2R_coeff=sr_config["L2R_coeff"],
                         optimizer="AdamW",
                         beta1=adam_config["beta1"],
                         beta2=adam_config["beta2"],
                         architecture=architecture)
        main(server_address=f"{server_ip}:{os.getenv('SERVER_PORT')}",
             client_name=str(os.getenv('CLIENT_NAME')),
             model=model, architecture=architecture)
        torch.cuda.empty_cache()
        sleep(10)
