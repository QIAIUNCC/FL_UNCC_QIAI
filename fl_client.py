import os
import flwr as fl
import pytorch_lightning as pl
from collections import OrderedDict
import torch
from dotenv import load_dotenv
from pytorch_lightning.callbacks import EarlyStopping
from fl_config import log_results, get_dataloaders
from hyperparameters import get_vit_config
from models.resnet import ResNet
from models.simple_vit import ViT
from utils.data_handler import get_datasets_classes
from utils.utils import set_seed, get_img_transformation, get_hyperparameters


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, train_loader, val_loader, test_loader, client_name, architecture):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.client_name = client_name,
        self.architecture = architecture

    def get_parameters(self, config):
        """
        Return the current local model parameters
        :param config:
        :return:
        """
        return _get_parameters(self.net.model)

    def set_parameters(self, parameters, config):
        _set_parameters(self.net.model, parameters)

    def fit(self, parameters, config):
        """
        Receive model parameters from the server, train the model parameters on the local data,
        and return the (updated) model parameters to the server
        :param parameters:
        :param config: dictionary contains the fit configuration
        :return: local model's parameters, length train data,
        """
        set_seed()
        self.set_parameters(parameters, config)
        early_stopping = EarlyStopping(monitor=config["monitor"], patience=config["patience"], verbose=False,
                                       mode=config["mode"])
        trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=config["max_epochs"],
                             callbacks=[early_stopping],
                             logger=False,
                             enable_checkpointing=False,
                             # log_every_n_steps=config["log_n_steps"],
                             )
        trainer.fit(model=self.net, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)

        return self.get_parameters(self.net.model), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        """
        Receive model parameters from the server, evaluate the model parameters on the local data,
        and return the evaluation result to the server
        :param parameters:
        :param config:
        :return:
        """
        self.set_parameters(parameters, config)
        trainer = pl.Trainer(accelerator='gpu', devices=[0], log_every_n_steps=1)
        print("============================")
        test_results = trainer.test(self.net, self.test_loader, verbose=True)

        loss = test_results[0]["test_loss"]

        # log_results(classes=self.net.hparams.classes,
        #             results=test_results,
        #             client_name=self.client_name,
        #             architecture=self.architecture,
        #             config=config)
        # print("================", self.client_name, "==============")
        # print("f1:", test_results[0]["test_f1"])
        # print("auc:", test_results[0]["test_auc"])
        # print("loss:", test_results[0]["test_loss"])
        # logging the validation for hyperparameter-tuning
        # val_results = trainer.test(self.net, self.val_loader, verbose=False)
        # log_results(classes=self.net.hparams.classes,
        #             results=val_results,
        #             client_name=self.client_name,
        #             architecture=self.architecture,
        #             config=config,
        #             log_suffix="val_log")

        return float(loss), len(self.test_loader), test_results[0]


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    # pass
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


load_dotenv(dotenv_path="../../data/.env")
DATASET_PATH = os.getenv('DATASET_PATH')
batch_size = 64
kermany_classes, srinivasan_classes, oct500_classes = get_datasets_classes()
architecture = "resnet18"


def client_fn_ViT(cid: str) -> FlowerClient:
    """Creates a FlowerClient instance on demand
    Create a Flower client representing a single organization
    """
    set_seed()
    # Load model
    train_loader, val_loader, test_loader, classes = get_dataloaders(cid=cid,
                                                                     dataset_path=DATASET_PATH,
                                                                     batch_size=batch_size,
                                                                     kermany_classes=kermany_classes,
                                                                     srinivasan_classes=srinivasan_classes,
                                                                     oct500_classes=oct500_classes,
                                                                     img_transforms=get_img_transformation(),
                                                                     limit_kermany=False
                                                                     )

    vit_config = get_vit_config()
    client_name = "kermany"
    if cid == "1":
        client_name = "srinivasan"
    elif cid == "2":
        client_name = "oct-500"

    model = ViT(classes=classes, lr=vit_config["lr"],
                weight_decay=vit_config["weight_decay"],
                **vit_config["model_config"])
    # Create a  single Flower client representing a single organization
    return FlowerClient(net=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        client_name=client_name, architecture="ViT")


def client_fn_ResNet(cid: str) -> FlowerClient:
    """Creates a FlowerClient instance on demand
    Create a Flower client representing a single organization
    """
    set_seed()
    # Load model
    train_loader, val_loader, test_loader, classes = get_dataloaders(cid=cid,
                                                                     dataset_path=DATASET_PATH,
                                                                     batch_size=batch_size,
                                                                     kermany_classes=kermany_classes,
                                                                     srinivasan_classes=srinivasan_classes,
                                                                     oct500_classes=oct500_classes,
                                                                     img_transforms=get_img_transformation(),
                                                                     limit_kermany=False
                                                                     )

    resnet_config = get_hyperparameters(model_architecture=architecture, client_name=os.getenv('CLIENT_NAME'))

    client_name = "kermany"
    if cid == "1":
        client_name = "srinivasan"
    elif cid == "2":
        client_name = "oct-500"

    model = ResNet(classes=classes,
                   lr=resnet_config["lr"],
                   weight_decay=resnet_config["wd"],
                   optimizer="AdamW",
                   beta1=resnet_config["beta1"],
                   beta2=resnet_config["beta2"],
                   architecture=architecture)
    # Create a  single Flower client representing a single organization
    return FlowerClient(net=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        client_name=client_name, architecture=architecture)


def main(server_address, model, architecture="resnet18", client_name="Kermany", cid="0") -> None:
    if client_name == "Srinivasan":
        cid = "1"
    elif client_name == "OCT-500":
        cid = "2"

    train_loader, val_loader, test_loader, classes = get_dataloaders(cid=cid,
                                                                     dataset_path=DATASET_PATH,
                                                                     batch_size=batch_size,
                                                                     kermany_classes=kermany_classes,
                                                                     srinivasan_classes=srinivasan_classes,
                                                                     oct500_classes=oct500_classes,
                                                                     img_transforms=get_img_transformation(),
                                                                     limit_kermany=False
                                                                     )
    # Flower client
    client = FlowerClient(net=model,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          test_loader=test_loader,
                          client_name=client_name, architecture=architecture)

    fl.client.start_numpy_client(server_address=server_address, client=client)
