from time import sleep
import torch
from dotenv import load_dotenv

from fl_client import main
import os
from models.resnet import ResNet
from utils.utils import set_seed, get_hyperparameters


if __name__ == "__main__":
    set_seed(10)
    classes = [("NORMAL", 0),
               ("AMD", 1)]
    architecture = "resnet18"
    load_dotenv(dotenv_path="../../data/.env")
    server_ip = os.getenv('SERVER_IP')
    resnet_config = get_hyperparameters(model_architecture=architecture, client_name=os.getenv('CLIENT_NAME'))
    # Model and data
    for i in range(1, 11):
        model = ResNet(classes=classes,
                       lr=resnet_config["lr"],
                       weight_decay=resnet_config["wd"],
                       optimizer="AdamW",
                       beta1=resnet_config["beta1"],
                       beta2=resnet_config["beta2"],
                       architecture=architecture)
        main(server_address=f"{server_ip}:{os.getenv('SERVER_PORT')}",
             client_name=str(os.getenv('CLIENT_NAME')),
             model=model, architecture=architecture)

        torch.cuda.empty_cache()
        sleep(10)
