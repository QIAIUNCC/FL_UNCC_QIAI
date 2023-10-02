import os
from time import sleep
import flwr as fl
import torch
from dotenv import load_dotenv
from FedAP.vit_APFL import ViTAP
from fl_strategy import FedAvgStrategy
from hyperparameters import get_AP_config, get_centralized_AdamW_ViT_parameters
from utils.utils import set_seed, weighted_average


def main(net, server_port) -> None:
    # Define strategy
    strategy = FedAvgStrategy(
        net=net,
        on_fit_config_fn=fit_config,  # The fit_config function we defined earlier
        on_evaluate_config_fn=eval_config,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=3,  # Never sample less than num_clients for training
        min_evaluate_clients=3,  # Never sample less than num_clients for evaluation
        # # Minimum number of clients that need to be connected to the server before a training round can start
        min_available_clients=3,
        # fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address=f"0.0.0.0:{server_port}",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "max_epochs": 5,
        "patience": 20,
        "monitor": "val_loss",
        "mode": "min",
        "share_batch_norm": True,
        "clients": 3,
        "train_batch_size": 64,
        "log_n_steps": 1,
    }
    return config


def eval_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    config = {
        "max_epochs": 5,
        "batch_size": 1,
        "current_round": server_round,
        "max_round": 1,
        "clients": 3,
        "train_batch_size": 64
    }
    return config


def simulation_main(net, client_fn) -> None:
    # Create FedAvg strategy
    strategy = FedAvgStrategy(
        net=net,
        on_fit_config_fn=fit_config,  # The fit_config function we defined earlier
        on_evaluate_config_fn=eval_config,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1,  # Sample 50% of available clients for evaluation
        # min_fit_clients=1,  # Never sample less than num_clients for training
        # min_evaluate_clients=1,  # Never sample less than num_clients for evaluation
        # # Minimum number of clients that need to be connected to the server before a training round can start
        # min_available_clients=1,
        # fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources={"num_gpus": 1},
    )


if __name__ == "__main__":
    set_seed(10)
    NUM_CLIENTS = 3
    classes = [("NORMAL", 0),
               ("AMD", 1)]
    vit_config = get_centralized_AdamW_ViT_parameters()
    ap_config = get_AP_config()
    load_dotenv(dotenv_path="../../data/.env")
    server_port = os.getenv('SERVER_PORT')
    architecture = "ViT"
    # Model and data
    for i in range(1, 11):
        model = ViTAP(classes=classes,
                      lr=vit_config["lr"],
                      weight_decay=vit_config["wd"],
                      optimizer="AdamW",
                      beta1=vit_config["beta1"],
                      beta2=vit_config["beta2"],
                      alpha=ap_config["alpha"],
                      )
        # simulation_main(net=model, client_fn=client_fn_ResNet)
        main(model, server_port)
        torch.cuda.empty_cache()
        sleep(1)
