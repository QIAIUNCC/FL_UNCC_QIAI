import torch
import torch.nn.functional as F

from models.resnet import ResNet


class ResNetProx(ResNet):

    def __init__(self, mu=1, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.server_model = None
        self.mean_log_keys = ["loss", "ce_loss", "fed_prox_reg"]

    def _calculate_loss(self, batch):
        imgs, labels = batch["img"], batch["label"]
        preds = self.forward(imgs)
        ce_loss = F.cross_entropy(preds, labels)

        fed_prox_reg = torch.zeros_like(ce_loss)
        if self.server_model is not None:
            curr_param = torch.cat([param.view(-1) for param in self.model.parameters()])
            serv_param = torch.cat([param.view(-1) for param in self.server_model.parameters()]).to(self.device)
            fed_prox_reg = torch.norm(curr_param - serv_param) ** 2

        loss = ce_loss + self.hparams.mu / 2 * fed_prox_reg
        return {"loss": loss, "preds": preds.argmax(dim=-1), "labels": labels, "ce_loss": ce_loss, "fed_prox_reg": fed_prox_reg}