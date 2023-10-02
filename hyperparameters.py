def get_kermany_AdamW_ResNet_parameters():
    return {
        "wd": 1e-6,
        "lr": 5e-4,
        "beta2": 0.93,
        "beta1": 0.88,
        "batch_size": 64,
    }


def get_srinivasan_AdamW_ResNet_parameters():
    return {
        "wd": 4e-5,
        "lr": 4e-5,
        "beta2": 0.94,
        "beta1": 0.83,
        "batch_size": 64,
    }


def get_oct500_AdamW_ResNet_parameters():
    return {
        "wd": 4e-5,
        "lr": 4e-5,
        "beta2": 0.94,
        "beta1": 0.83,
        "batch_size": 64,
    }


def get_centralized_AdamW_ResNet_parameters():
    return {
        "wd": 5e-5,
        "lr": 6e-4,
        "beta2": 0.92,
        "beta1": 0.9,
        "batch_size": 64,
    }


def get_kermany_AdamW_ViT_parameters():
    return {
        "wd": 2e-5,
        "lr": 1e-4,
        "beta2": 0.98,
        "beta1": 0.83,
        "batch_size": 64,
    }


def get_srinivasan_AdamW_ViT_parameters():
    return {
        "wd": 2e-5,
        "lr": 4e-4,
        "beta2": 0.9,
        "beta1": 0.81,
        "batch_size": 64,
    }


def get_oct500_AdamW_ViT_parameters():
    return {
        "wd": 4e-5,
        "lr": 4e-5,
        "beta2": 0.94,
        "beta1": 0.83,
        "batch_size": 64,
    }


def get_centralized_AdamW_ViT_parameters():
    return {
        "wd": 3e-6,
        "lr": 2e-4,
        "beta2": 0.96,
        "beta1": 0.8,
        "batch_size": 64,
    }


def get_vit_config():
    return {"image_size": 128,
            "patch_size": 8,
            "num_classes": 2,
            "dim": 256,
            "depth": 6,
            "heads": 8,
            "mlp_dim": 512,
            }


def get_MRI_config():
    return {
        "mu": 100,
        "lr_drop": 80,
        "lr_gamma": 0.1,
        "batch_size": 64,
    }


def get_AP_config():
    return {
        "alpha": 0.5,
        "batch_size": 64,
    }


def get_Prox_config():
    "You might want to tune mu from {0.001, 0.01, 0.1, 0.5, 1}"
    return {
        "mu": 1,
        "batch_size": 64,
    }


def get_SR_config():
    return {
        "batch_size": 64,
        "CMI_coeff": 0.001,
        "L2R_coeff": 0.01,
        "z_dim": 512,
        "num_samples": 20}

