from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Optional
import flwr as fl
import numpy as np
import torch
from flwr.common import FitRes, Scalar, Parameters, parameters_to_ndarrays
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedAvgStrategy(FedAvg):
    def __init__(self,
                 *,
                 net=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.net = net

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.net.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.model.load_state_dict(state_dict, strict=True)
            if server_round == 10:
                # Save the model
                torch.save(self.net.model.state_dict(), f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics


class FedMRIStrategy(FedAvgStrategy):
    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        params = [param for _, param in self.net.state_dict().items()] * 4
        # 1: server, 3 clients
        return fl.common.ndarrays_to_parameters(params)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
        params = aggregated_ndarrays
        for _, fit_res in results:
            params += parameters_to_ndarrays(fit_res.parameters)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            # aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.net.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.model.load_state_dict(state_dict, strict=True)
            # Save the model
            if server_round == 10:
                torch.save(self.net.model.state_dict(), f"model_round_{server_round}.pth")

        return fl.common.ndarrays_to_parameters(params), aggregated_metrics
