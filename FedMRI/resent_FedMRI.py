import torch
import torch.nn.functional as F
from models.resnet import ResNet


class ResNetMRI(ResNet):
    def __init__(self, mu=0.1, lr_drop=80, lr_gamma=0.1, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.server_param = None
        self.mean_log_keys = ["loss", "ce_loss", "con_loss"]
        self.other_models = []
        self.server_model = None

    def _calculate_contrastive_loss(self):
        curr_param = torch.cat([param.view(-1) for param in self.model.parameters()])
        serv_param = torch.cat([param.view(-1) for param in self.server_model.parameters()]).to(self.device)
        norm_pos = torch.norm(curr_param - serv_param, p=1)
        norm_neg = []
        for model in self.other_models:
            other_param = torch.cat([param.view(-1) for param in model.parameters()]).to(self.device)
            norm_neg.append(torch.norm(curr_param - other_param, p=1))
        l_con = torch.div(norm_pos, sum(norm_neg) + 1e-14)
        return l_con

    def _calculate_loss(self, batch):
        imgs, labels = batch["img"], batch["label"]
        preds = self.forward(imgs)
        ce_loss = F.cross_entropy(preds, labels)
        con_loss = self._calculate_contrastive_loss() if self.server_model else torch.zeros_like(ce_loss)
        loss = ce_loss + self.hparams.mu * con_loss
        return {"loss": loss, "preds": preds.argmax(dim=-1), "labels": labels, "ce_loss": ce_loss, "con_loss": con_loss}
