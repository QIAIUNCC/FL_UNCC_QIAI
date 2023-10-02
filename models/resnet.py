import torchvision
from models.base import BaseNet


class ResNet(BaseNet):
    def __init__(self, architecture, **kwargs):
        """
        kwargs:{
            classes, lr, weight_decay, architecture, optimizer, momentum, dampening, beta1, beta2
        }
        """
        super().__init__(**kwargs)
        self.model = torchvision.models.resnet18(num_classes=len(self.hparams.classes))
        if architecture == "resnet50":
            self.model = torchvision.models.resnet50(num_classes=len(self.hparams.classes))
        elif architecture == "resnet101":
            self.model = torchvision.models.resnet101(num_classes=len(self.hparams.classes))
        elif architecture == "resnet152":
            self.model = torchvision.models.resnet152(num_classes=len(self.hparams.classes))
        self.save_hyperparameters()
