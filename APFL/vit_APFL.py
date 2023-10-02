import copy
import numpy as np
import torch.nn.functional as F
from torch import optim

from models.resnet import ResNet
from models.simple_vit import ViT


class ViTAP(ViT):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.automatic_optimization = False  # Set manual optimization
        self.model_per = copy.deepcopy(self.model)
        self.mean_log_keys = ["loss", "ce_w_loss", "ce_v_loss"]

    def forward(self, x):
        return self.model(x), self.model_per(x)

    def training_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        # Retrieve the optimizers
        opt_w, opt_v = self.optimizers()
        # Manually backpropagate the losses and step the optimizers
        # Compute gradients for the current step
        self.manual_backward(output["ce_w_loss"])

        # Compute gradients for the current step
        self.manual_backward(output["ce_v_loss"])

        self.alpha_update()
        # Update parameters based on current gradients
        opt_w.step()
        # Clear gradients for the next step
        opt_w.zero_grad()
        # Update parameters based on current gradients
        opt_v.step()
        # Clear gradients for the next step
        opt_v.zero_grad()

        self.step_output["train"].append(output)
        return output

    def _calculate_loss(self, batch):
        imgs, labels = batch["img"], batch["label"]
        w_preds, v_preds = self.forward(imgs)
        ce_w_loss = F.cross_entropy(w_preds, labels)
        ce_v_loss = F.cross_entropy(v_preds, labels)
        return {"loss": ce_w_loss + ce_v_loss, "preds": v_preds.argmax(dim=-1), "labels": labels,
                "ce_w_loss": ce_w_loss, "ce_v_loss": ce_v_loss}

    def alpha_update(self):
        grad_alpha = 0
        for l_params, p_params in zip(self.model.parameters(), self.model_per.parameters()):
            dif = p_params.data - l_params.data
            grad = self.hparams.alpha * p_params.grad.data + (1 - self.hparams.alpha) * l_params.grad.data
            grad_alpha += dif.view(-1).dot(grad.view(-1))

        grad_alpha += 0.02 * self.hparams.alpha
        self.hparams.alpha = self.hparams.alpha - self.hparams.lr * grad_alpha
        self.hparams.alpha = np.clip(self.hparams.alpha.item(), 0.0, 1.0)

    def update_personalize_model_param(self):
        for lp, p in zip(self.model_per.parameters(), self.model.parameters()):
            lp.data = (1 - self.hparams.alpha) * p + self.hparams.alpha * lp

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            return [optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
                                betas=(self.hparams.beta1, self.hparams.beta2)),
                    optim.AdamW(self.model_per.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay,
                                betas=(self.hparams.beta1, self.hparams.beta2))]
        if self.hparams.optimizer == "Adam":
            return [optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay),
                    optim.Adam(self.model_per.parameters(), lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)]

        if self.hparams.optimizer == "SGD":
            return [optim.SGD(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
                              dampening=self.hparams.dampening, momentum=self.hparams.momentum),
                    optim.SGD(self.model_per.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
                              dampening=self.hparams.dampening, momentum=self.hparams.momentum)]
