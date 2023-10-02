import copy
import os
from time import sleep
from dotenv import load_dotenv
from FedMRI.resent_FedMRI import ResNetMRI
from fl_client import FlowerClient, _set_parameters
import flwr as fl
from fl_config import get_dataloaders
from hyperparameters import get_MRI_config
from utils.data_handler import get_datasets_classes
from utils.utils import get_img_transformation, set_seed, get_hyperparameters


class FlowerClientMRI(FlowerClient):

    def set_parameters(self, parameters, config):
        """
        :param parameters: a list of parameters sent by the server
        :param config:
        :return:
        """
        if config["current_round"] > 1:
            c = len(parameters) // (config["clients"] + 1)  # number of clients + 1 server
            self.net.server_model = copy.deepcopy(self.net.model)
            _set_parameters(self.net.server_model, parameters[0:c])
            self.net.other_models.clear()
            for i in range(1, len(parameters) // c):
                model = copy.deepcopy(self.net.model)
                _set_parameters(model, parameters[c * i:c * (i + 1)])
                self.net.other_models.append(model)


load_dotenv(dotenv_path="../../data/.env")
mri_config = get_MRI_config()
architecture = "resnet18"
client_name = str(os.getenv('CLIENT_NAME'))
adam_config = get_hyperparameters(model_architecture=architecture, client_name=client_name)
DATASET_PATH = os.getenv('DATASET_PATH')
kermany_classes, srinivasan_classes, oct500_classes = get_datasets_classes()
cid = "0"
if client_name == "Srinivasan":
    cid = "1"
elif client_name == "OCT-500":
    cid = "2"
train_loader, val_loader, test_loader, classes = get_dataloaders(cid=cid,
                                                                 dataset_path=DATASET_PATH,
                                                                 batch_size=mri_config["batch_size"],
                                                                 kermany_classes=kermany_classes,
                                                                 srinivasan_classes=srinivasan_classes,
                                                                 oct500_classes=oct500_classes,
                                                                 img_transforms=get_img_transformation(),
                                                                 limit_kermany=False
                                                                 )


def client_fn(cid: str) -> FlowerClient:
    """Creates a FlowerClient instance on demand
    Create a Flower client representing a single organization
    """
    set_seed(10)
    return FlowerClientMRI(model, train_loader, val_loader, test_loader, client_name=client_name,
                           architecture=architecture)


if __name__ == "__main__":
    set_seed(10)
    server_ip = os.getenv('SERVER_IP')

    # Flower client
    for i in range(1, 11):
        model = ResNetMRI(classes=classes,
                          lr=adam_config["lr"],
                          weight_decay=adam_config["wd"],
                          optimizer="AdamW",
                          beta1=adam_config["beta1"],
                          beta2=adam_config["beta2"],
                          mu=mri_config["mu"],
                          lr_drop=mri_config["lr_drop"],
                          lr_gamma=mri_config["lr_gamma"],
                          architecture=architecture
                          )
        client = FlowerClientMRI(model, train_loader, val_loader, test_loader, client_name=client_name,
                                 architecture=architecture)
        fl.client.start_numpy_client(server_address=f"{server_ip}:{os.getenv('SERVER_PORT')}",
                                     client=client)
        sleep(10)
