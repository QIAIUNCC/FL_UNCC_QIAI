import copy
from time import sleep
import torch
from dotenv import load_dotenv
import flwr as fl
from FedProx.resent_FedProx import ResNetProx
from fl_client import FlowerClient
import os
from fl_config import get_dataloaders
from hyperparameters import get_Prox_config
from utils.data_handler import get_datasets_classes
from utils.utils import get_img_transformation, set_seed, get_hyperparameters


class FlowerClientProx(FlowerClient):

    def fit(self, parameters, config):
        """
        Receive model parameters from the server, train the model parameters on the local data,
        and return the (updated) model parameters to the server
        :param parameters:
        :param config: dictionary contains the fit configuration
        :return: local model's parameters, length train data,
        """
        if config["current_round"] > 1:
            self.net.server_model = copy.deepcopy(self.net.model)
        return super(FlowerClientProx, self).fit(parameters, config)


set_seed(10)
load_dotenv(dotenv_path="../../data/.env")
DATASET_PATH = os.getenv('DATASET_PATH')
prox_config = get_Prox_config()
client_name = str(os.getenv('CLIENT_NAME'))
architecture = "resnet18"
adam_config = get_hyperparameters(model_architecture=architecture, client_name=client_name)
kermany_classes, srinivasan_classes, oct500_classes = get_datasets_classes()
cid = "0"
if client_name == "Srinivasan":
    cid = "1"
elif client_name == "OCT-500":
    cid = "2"
train_loader, val_loader, test_loader, classes = get_dataloaders(cid=cid,
                                                                 dataset_path=DATASET_PATH,
                                                                 batch_size=prox_config["batch_size"],
                                                                 kermany_classes=kermany_classes,
                                                                 srinivasan_classes=srinivasan_classes,
                                                                 oct500_classes=oct500_classes,
                                                                 img_transforms=get_img_transformation(),
                                                                 limit_kermany=False
                                                                 )




def client_fn_ResNetProx(cid: str) -> FlowerClientProx:
    """Creates a FlowerClient instance on demand
    Create a Flower client representing a single organization
    """
    set_seed(10)
    train_loader, val_loader, test_loader, classes = get_dataloaders(cid=cid,
                                                                     dataset_path=DATASET_PATH,
                                                                     batch_size=prox_config["batch_size"],
                                                                     kermany_classes=kermany_classes,
                                                                     srinivasan_classes=srinivasan_classes,
                                                                     oct500_classes=oct500_classes,
                                                                     img_transforms=get_img_transformation(),
                                                                     limit_kermany=False
                                                                     )
    return FlowerClientProx(model, train_loader, val_loader, test_loader,
                            client_name=client_name, architecture=architecture)


if __name__ == "__main__":
    set_seed(10)
    server_ip = os.getenv('SERVER_IP')
    # Model and data
    for i in range(1, 6):
        model = ResNetProx(classes=classes,
                           lr=adam_config["lr"],
                           weight_decay=adam_config["wd"],
                           optimizer="AdamW",
                           beta1=adam_config["beta1"],
                           beta2=adam_config["beta2"],
                           mu=prox_config["mu"],
                           architecture=architecture)
        client = FlowerClientProx(model, train_loader, val_loader, test_loader,
                                  client_name=client_name,
                                  architecture=architecture)
        fl.client.start_numpy_client(server_address=f"{server_ip}:{os.getenv('SERVER_PORT')}",
                                     client=client)
        torch.cuda.empty_cache()
        sleep(10)
