import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_curve, auc
from torch import optim
import torch.nn.functional as F
from torchmetrics import F1Score, AUROC
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification.accuracy import MulticlassAccuracy


class BaseNet(pl.LightningModule):
    def __init__(self, classes, lr, weight_decay=0, momentum=0, dampening=0, optimizer="SGD", beta1=0.9, beta2=0.999):
        """

        :param classes (tuple(str, int)): list of tuples, each tuple consists of class name and class index
        :param lr (float): learning rate
        :param weight_decay (float): weight decay of optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        task = "binary" if len(classes) == 2 else "multiclass"
        self.metrics_list = ["accuracy", "precision", "f1", "auc"]
        self.sessions = ["train", "val", "test"]

        self.train_ac = MulticlassAccuracy(task=task, num_classes=len(classes), average=None)
        self.val_ac = MulticlassAccuracy(task=task, num_classes=len(classes), average=None)
        self.test_ac = MulticlassAccuracy(task=task, num_classes=len(classes), average=None)

        self.train_p = MulticlassPrecision(task=task, num_classes=len(classes), average=None)
        self.val_p = MulticlassPrecision(task=task, num_classes=len(classes), average=None)
        self.test_p = MulticlassPrecision(task=task, num_classes=len(classes), average=None)

        self.train_f1 = F1Score(task=task, num_classes=len(classes))
        self.val_f1 = F1Score(task=task, num_classes=len(classes))
        self.test_f1 = F1Score(task=task, num_classes=len(classes))

        self.train_auc = AUROC(task=task, num_classes=len(classes))
        self.val_auc = AUROC(task=task, num_classes=len(classes))
        self.test_auc = AUROC(task=task, num_classes=len(classes))

        self.metrics = {"train": [self.train_ac, self.train_p, self.train_f1, self.train_auc],
                        "val": [self.val_ac, self.val_p, self.val_f1, self.val_auc],
                        "test": [self.test_ac, self.test_p, self.test_f1, self.test_auc],
                        }
        self.step_output = {"train": [], "val": [], "test": []}
        self.mean_log_keys = ["loss"]

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            return optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
                               betas=(self.hparams.beta1, self.hparams.beta2))
        if self.hparams.optimizer == "Adam":
            return optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.optimizer == "SGD":
            return optim.SGD(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
                             dampening=self.hparams.dampening, momentum=self.hparams.momentum)

    def forward(self, x):
        return self.model(x)

    def _calculate_loss(self, batch):
        imgs, labels = batch["img"], batch["label"]
        preds = self.forward(imgs)
        loss = F.cross_entropy(preds, labels)
        preds = preds.argmax(dim=-1) if len(self.hparams.classes) == 2 else preds
        return {"loss": loss, "preds": preds, "labels": labels}

    def training_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.step_output["train"].append(output)
        return output

    # def on_train_batch_end(self, batch, batch_idx, dataloader_idx):
    #     preds = batch["preds"]
    #     labels = batch["labels"]
    #     self.update_metrics(session="train", preds=preds, labels=labels)
    #     return batch

    def on_train_epoch_end(self):
        self.stack_update(session="train")

    def validation_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.step_output["val"].append(output)
        return output

    # def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
    #     preds = batch_parts["preds"]
    #     labels = batch_parts["labels"]
    #     self.update_metrics(session="val", preds=preds, labels=labels)
    #     return batch_parts

    def on_validation_epoch_end(self):
        self.stack_update(session="val")

    def test_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.step_output["test"].append(output)
        return output

    # def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
    #     preds = batch_parts["preds"]
    #     labels = batch_parts["labels"]
    #     self.update_metrics(session="test", preds=preds, labels=labels)
    #     return batch_parts

    def on_test_epoch_end(self, ):
        self.stack_update(session="test")

    def update_metrics(self, session, preds, labels):
        for metric in self.metrics[session]:
            metric.update(preds, labels)

    def stack_update(self, session):
        all_preds = torch.cat([out["preds"] for out in self.step_output[session]])
        all_labels = torch.cat([out["labels"] for out in self.step_output[session]])
        log = {}
        for key in self.mean_log_keys:
            log[f"{session}_{key}"] = torch.stack([out[key] for out in self.step_output[session]]).mean()

        self.update_metrics(session=session, preds=all_preds, labels=all_labels)
        res = self.compute_metrics(session=session)
        self.add_log(session, res, log)
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True, logger=True)
        self.restart_metrics(session=session)

        return all_preds, all_labels

    def compute_metrics(self, session):
        res = {}
        for metric, metric_name in zip(self.metrics[session], self.metrics_list):
            res[metric_name] = metric.compute()
        return res

    def restart_metrics(self, session):
        for metric in self.metrics[session]:
            metric.reset()
        self.step_output[session].clear()  # free memory

    def add_log(self, session, res, log):
        if "auc" in self.metrics_list:
            log[session + "_auc"] = res["auc"]
        if "f1" in self.metrics_list:
            log[session + "_f1"] = res["f1"]

        if "precision" in self.metrics_list:
            for key, idx in self.hparams.classes:
                log[session + "_precision_" + key] = res["precision"][idx]
        if "accuracy" in self.metrics_list:
            for key, idx in self.hparams.classes:
                log[session + "_accuracy_" + key] = res["accuracy"][idx]
