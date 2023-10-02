import copy

from torch import nn
from torch.nn import functional as F
import torch
import torch.nn
import torch.distributions as distributions

from models.simple_vit import ViT


class ViTSR(ViT):
    def __init__(self, L2R_coeff=0.01, CMI_coeff=0.001, num_samples=20, z_dim=256, **kwargs):

        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.encoder = copy.deepcopy(self.model)
        self.encoder.linear_head = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, 2 * z_dim)
        )
        self.num_classes = len(self.hparams.classes)
        self.cls = nn.Linear(z_dim, self.num_classes)
        self.r_mu = nn.Parameter(torch.zeros(self.num_classes, z_dim))
        self.r_sigma = nn.Parameter(torch.ones(self.num_classes, z_dim))
        self.C = nn.Parameter(torch.ones([]))
        self.model = nn.Sequential(self.encoder, self.cls)
        self.mean_log_keys = ["loss", "ce_loss", "regL2R", "regCMI", "regNegEnt"]

    def featurize(self, x, num_samples=1, return_dist=False):
        z_params = self.encoder(x)
        z_mu = z_params[:, :self.hparams.z_dim]
        z_sigma = F.softplus(z_params[:, self.hparams.z_dim:])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample([num_samples]).view([-1, self.hparams.z_dim])
        if return_dist:
            return z, (z_mu, z_sigma)
        else:
            return z

    def forward(self, x):
        if self.training:
            z = self.featurize(x)
            return self.cls(z)
        else:
            z = self.featurize(x, num_samples=self.num_samples)
            preds = torch.softmax(self.cls(z), dim=1)
            preds = preds.view([self.hparams.num_samples, -1, self.num_classes]).mean(0)
            return torch.log(preds)

    def _calculate_loss(self, batch):
        imgs, labels = batch["img"], batch["label"]
        z, (z_mu, z_sigma) = self.featurize(imgs, return_dist=True)
        preds = self.cls(z)
        ce_loss = F.cross_entropy(preds, labels)
        # https://github.com/atuannguyen/FedSR/blob/main/src/methods/FedSR.py
        obj = ce_loss

        regL2R = torch.zeros_like(obj)
        regCMI = torch.zeros_like(obj)

        if self.hparams.L2R_coeff != 0.0:
            regL2R = z.norm(dim=1).mean()
            obj = obj + self.hparams.L2R_coeff * regL2R

        if self.hparams.CMI_coeff != 0.0:
            r_sigma_softplus = F.softplus(self.r_sigma)
            r_mu = self.r_mu[labels]
            r_sigma = r_sigma_softplus[labels]
            z_mu_scaled = z_mu * self.C
            z_sigma_scaled = z_sigma * self.C
            regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                     (z_sigma_scaled ** 2 + (z_mu_scaled - r_mu) ** 2) / (2 * r_sigma ** 2) - 0.5
            regCMI = regCMI.sum(1).mean()
            obj = obj + self.hparams.CMI_coeff * regCMI

        z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
        mix_coeff = distributions.categorical.Categorical(imgs.new_ones(imgs.shape[0]))
        mixture = distributions.mixture_same_family.MixtureSameFamily(mix_coeff, z_dist)
        log_prob = mixture.log_prob(z)
        regNegEnt = log_prob.mean()

        return {"loss": obj, "preds": preds.argmax(dim=-1), "labels": labels, "ce_loss": ce_loss,
                "regL2R": regL2R, "regCMI": regCMI, "regNegEnt": regNegEnt}
