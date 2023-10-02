from time import sleep
import torch
from dotenv import load_dotenv

from fl_client import main
import os
from models.simple_vit import ViT
from utils.utils import set_seed, get_hyperparameters

if __name__ == "__main__":
    set_seed(10)
    classes = [("NORMAL", 0),
               ("AMD", 1)]
    adam_config = get_hyperparameters(model_architecture="ViT", client_name=os.getenv('CLIENT_NAME'))
    load_dotenv(dotenv_path="../../data/.env")
    server_ip = os.getenv('SERVER_IP')
    # Model and data
    for i in range(1, 11):
        model = ViT(classes=classes,
                    lr=adam_config["lr"],
                    weight_decay=adam_config["wd"],
                    optimizer="AdamW",
                    beta1=adam_config["beta1"],
                    beta2=adam_config["beta2"],
                    )
        main(server_address=f"{server_ip}:{os.getenv('SERVER_PORT')}",
             client_name=str(os.getenv('CLIENT_NAME')),
             model=model, architecture="ViT")

        torch.cuda.empty_cache()
        sleep(10)
