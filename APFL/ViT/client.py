import copy
from time import sleep
from dotenv import load_dotenv
import flwr as fl
from FedAP.vit_APFL import ViTAP
from fl_client import FlowerClient
import os
from fl_config import get_dataloaders
from hyperparameters import get_AP_config
from utils.data_handler import get_datasets_classes
from utils.utils import set_seed, get_hyperparameters, get_img_transformation


class FlowerClientAP(FlowerClient):
    def fit(self, parameters, config):
        out = super().fit(parameters, config)
        self.net.update_personalize_model_param()
        return out

    def evaluate(self, parameters, config):
        out = super().evaluate(parameters, config)
        if config["current_round"] == 10:
            torch.save(self.net.model_per.state_dict(), f"client_personalized_model_10.pth")


load_dotenv(dotenv_path="../../data/.env")
ap_config = get_AP_config()
architecture = "ViT"
client_name = str(os.getenv('CLIENT_NAME'))
vit_config = get_hyperparameters(model_architecture=architecture, client_name=client_name)
DATASET_PATH = os.getenv('DATASET_PATH')
kermany_classes, srinivasan_classes, oct500_classes = get_datasets_classes()
cid = "0"
if client_name == "Srinivasan":
    cid = "1"
elif client_name == "OCT-500":
    cid = "2"
train_loader, val_loader, test_loader, classes = get_dataloaders(cid=cid,
                                                                 dataset_path=DATASET_PATH,
                                                                 batch_size=ap_config["batch_size"],
                                                                 kermany_classes=kermany_classes,
                                                                 srinivasan_classes=srinivasan_classes,
                                                                 oct500_classes=oct500_classes,
                                                                 img_transforms=get_img_transformation(),
                                                                 limit_kermany=False
                                                                 )
if __name__ == "__main__":
    set_seed(10)
    server_ip = os.getenv('SERVER_IP')

    # Flower client
    for i in range(1, 11):
        model = ViTAP(classes=classes,
                      lr=vit_config["lr"],
                      weight_decay=vit_config["wd"],
                      optimizer="AdamW",
                      beta1=vit_config["beta1"],
                      beta2=vit_config["beta2"],
                      alpha=ap_config["alpha"],
                      )
        client = FlowerClientAP(model, train_loader, val_loader, test_loader, client_name=client_name,
                                architecture=architecture)
        fl.client.start_numpy_client(server_address=f"{server_ip}:{os.getenv('SERVER_PORT')}",
                                     client=client)
        sleep(10)
