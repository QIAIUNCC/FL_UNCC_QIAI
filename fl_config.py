import json
import os
from torch.utils.data import DataLoader
import torch
from models.simple_vit import ViT
from utils.data_handler import get_oct500_datasets, get_data_loaders, get_srinivasan_datasets, get_kermany_datasets


def get_dataloaders(cid, dataset_path, batch_size, kermany_classes, srinivasan_classes, oct500_classes, img_transforms,
                    limit_kermany):
    train_loader = None
    val_loader = None
    test_loader = None
    classes = None
    if limit_kermany:
        oct500_dataset_train_6mm, oct500_dataset_val_6mm, oct500_dataset_test_6mm = \
            get_oct500_datasets(dataset_path + "/2/OCTA_6mm", oct500_classes, img_transformation=img_transforms)
        oct500_dataset_train_3mm, oct500_dataset_val_3mm, oct500_dataset_test_3mm = \
            get_oct500_datasets(dataset_path + "/2/OCTA_3mm", oct500_classes, img_transformation=img_transforms)
        oct500_dataset_train = torch.utils.data.ConcatDataset([oct500_dataset_train_6mm, oct500_dataset_train_3mm])
        oct500_dataset_val = torch.utils.data.ConcatDataset([oct500_dataset_val_6mm, oct500_dataset_val_3mm])
        oct500_dataset_test = torch.utils.data.ConcatDataset([oct500_dataset_test_6mm, oct500_dataset_test_3mm])

    if cid == "0":
        classes = kermany_classes
        kermany_dataset_train, kermany_dataset_val, kermany_dataset_test = get_kermany_datasets(
            dataset_path + "/0/train",
            dataset_path + "/0/test", kermany_classes, img_transformation=img_transforms, val_split=0.05,
        )
        if limit_kermany:
            kermany_dataset_train, _ = torch.utils.data.random_split(kermany_dataset_train, [len(oct500_dataset_train),
                                                                                           len(kermany_dataset_train) -
                                                                                           len(
                                                                                               oct500_dataset_train)],
                                                                    generator=torch.Generator().manual_seed(10))
            kermany_dataset_val, _ = torch.utils.data.random_split(kermany_dataset_val, [len(oct500_dataset_val),
                                                                                       len(kermany_dataset_val) - len(
                                                                                           oct500_dataset_val)],
                                                                  generator=torch.Generator().manual_seed(10))
            kermany_dataset_test, _ = torch.utils.data.random_split(kermany_dataset_test, [len(oct500_dataset_test),
                                                                                         len(kermany_dataset_test) - len(
                                                                                             oct500_dataset_test)],
                                                                   generator=torch.Generator().manual_seed(10))
        train_loader, val_loader = get_data_loaders(kermany_dataset_train, kermany_dataset_val, batch_size)
        test_loader = DataLoader(kermany_dataset_test, batch_size=1, shuffle=False,
                                 drop_last=False, pin_memory=True, num_workers=4)

    elif cid == "1":
        classes = srinivasan_classes
        srinivasan_dataset_train, srinivasan_dataset_val, srinivasan_dataset_test = get_srinivasan_datasets(
            train_path=dataset_path + "/1/train", test_path=dataset_path + "/1/test", classes=srinivasan_classes,
            img_transformation=img_transforms)
        train_loader, val_loader = get_data_loaders(srinivasan_dataset_train, srinivasan_dataset_val, batch_size)
        test_loader = DataLoader(srinivasan_dataset_test, batch_size=1, shuffle=False,
                                 drop_last=False, pin_memory=True, num_workers=4)
        print("srinivasan len:", len(train_loader))

    elif cid == "2":
        classes = oct500_classes
        oct500_dataset_train_6mm, oct500_dataset_val_6mm, oct500_dataset_test_6mm = \
            get_oct500_datasets(dataset_path + "/2/OCTA_6mm", oct500_classes, img_transformation=img_transforms)
        oct500_dataset_train_3mm, oct500_dataset_val_3mm, oct500_dataset_test_3mm = \
            get_oct500_datasets(dataset_path + "/2/OCTA_3mm", oct500_classes, img_transformation=img_transforms)

        oct500_dataset_train = torch.utils.data.ConcatDataset([oct500_dataset_train_6mm, oct500_dataset_train_3mm])
        oct500_dataset_val = torch.utils.data.ConcatDataset([oct500_dataset_val_6mm, oct500_dataset_val_3mm])
        oct500_dataset_test = torch.utils.data.ConcatDataset([oct500_dataset_test_6mm, oct500_dataset_test_3mm])

        train_loader, val_loader = get_data_loaders(oct500_dataset_train, oct500_dataset_val, batch_size)
        test_loader = DataLoader(oct500_dataset_test, batch_size=1, shuffle=False,
                                 drop_last=False, pin_memory=True, num_workers=4)
        print("OCT 500 len:", len(train_loader))

    return train_loader, val_loader, test_loader, classes


def read_last_results(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            last_data = f.readlines()[-1]
            last_data = last_data.replace("\'", "\"")
            res = json.loads(last_data)
        return res
    return False


def log_results(classes, results, client_name, architecture, config, log_suffix="", approach="FL"):
    result = {}
    metrics = ["accuracy", "precision"]
    for c in classes:
        for m in metrics:
            result[f"test_{m}_" + c[0]] = results[0][f"test_{m}_" + c[0]]

    result["f1_score"] = results[0]["test_f1"]
    result["auc"] = results[0]["test_auc"]
    result["loss"] = results[0]["test_loss"]
    dir_name = f"log_e{config['max_epochs']}/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    arch = architecture.replace(')', '').replace('(', '').replace(',', '').replace("'", "")
    log_name = f'{dir_name}/{client_name}_{approach}_{arch}_{config["train_batch_size"]}_{log_suffix}.txt'
    file_mode = "a" if os.path.exists(log_name) else "w"
    with open(log_name, file_mode) as f:
        f.write(f"============={config['current_round']}=======================")
        f.write('\n')
        f.write(str(result))
        f.write('\n')


def create_model(config):
    if config["model_architecture"] == "FedAvg_ViT":
        model = ViT(classes=config["classes"], lr=config["lr"], weight_decay=config["weight_decay"],
                    model_config=config["model_config"])

    elif config["model_architecture"] == "FedSR_ViT":
        model = ViT(classes=config["classes"], lr=config["lr"], weight_decay=config["weight_decay"],
                    model_config=config["model_config"])

    elif config["model_architecture"] == "FedProx_ViT":
        model = ViT(classes=config["classes"], lr=config["lr"], weight_decay=config["weight_decay"],
                    model_config=config["model_config"])

    elif config["model_architecture"] == "FedGain_ViT":
        model = ViT(classes=config["classes"], lr=config["lr"], weight_decay=config["weight_decay"],
                    model_config=config["model_config"])

    elif config["model_architecture"] == "FedGain_ViT":
        model = ViT(classes=config["classes"], lr=config["lr"], weight_decay=config["weight_decay"],
                    model_config=config["model_config"])

    return model


if __name__ == "__main__":
    out = read_last_results("FedAvg/ResNet/log_resnet50/srinivasan_FL_resnet50_64.txt")
    print(out)
