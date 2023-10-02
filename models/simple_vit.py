from vit_pytorch import SimpleViT

from hyperparameters import get_vit_config
from models.base import BaseNet


class ViT(BaseNet):
    def __init__(self,  **kwargs):
        """
        :param classes (tuple(str, int)): list of tuples, each tuple consists of class name and class index
        :param lr (float): learning rate
        :param weight_decay (float): weight decay of optimizer
        """
        super().__init__(**kwargs)
        self.model = SimpleViT(**get_vit_config())
