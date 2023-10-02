import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from data_handler import get_datasets_classes, get_kermany_datasets, get_srinivasan_datasets, get_oct500_datasets


# Define function to calculate the average histogram of a group of images
def avg_hist(dataloader):
    # Initialize the total histogram
    total_hist = np.zeros((256,))

    # Read each image in the directory
    for img_data in dataloader:
        img = cv2.imread(img_data["img_path"], cv2.IMREAD_GRAYSCALE)

        # Calculate normalized histogram
        hist_img = cv2.calcHist([img], [0], None, [256], [0, 256])
        cv2.normalize(hist_img, hist_img)

        # Add to the total histogram
        total_hist += np.squeeze(hist_img)  # squeeze out the unnecessary dimension

    # Average the total histogram
    avg_hist = total_hist / len(dataloader)

    return avg_hist

load_dotenv(dotenv_path="../data/.env")
server_port = os.getenv('DATASET_PATH')
DATASET_PATH = os.getenv('DATASET_PATH')
kermany_classes, srinivasan_classes, oct500_classes = get_datasets_classes()
NUM_WORKERS = 4

kermany_dataset_train, kermany_dataset_val, kermany_dataset_test = get_kermany_datasets(
        DATASET_PATH + "/0/train",
        DATASET_PATH + "/0/test",
        kermany_classes,
        img_transformation=None,
        val_split=0.05,
    )

kermany_dataset = torch.utils.data.ConcatDataset([kermany_dataset_train, kermany_dataset_val, kermany_dataset_test])

oct500_dataset_train_6mm, oct500_dataset_val_6mm, oct500_dataset_test_6mm = \
        get_oct500_datasets(DATASET_PATH + "/2/OCTA_6mm", oct500_classes, img_transformation=None)
oct500_dataset_train_3mm, oct500_dataset_val_3mm, oct500_dataset_test_3mm = \
        get_oct500_datasets(DATASET_PATH + "/2/OCTA_3mm", oct500_classes, img_transformation=None)

oct500_dataset_train = torch.utils.data.ConcatDataset([oct500_dataset_train_6mm, oct500_dataset_train_3mm])
oct500_dataset_val = torch.utils.data.ConcatDataset([oct500_dataset_val_6mm, oct500_dataset_val_3mm])
oct500_dataset_test = torch.utils.data.ConcatDataset([oct500_dataset_test_6mm, oct500_dataset_test_3mm])
oct500_dataset = torch.utils.data.ConcatDataset([oct500_dataset_train, oct500_dataset_val, oct500_dataset_test])

srinivasan_dataset_train, srinivasan_dataset_val, \
        srinivasan_dataset_test = get_srinivasan_datasets(DATASET_PATH + "/1/train",
                                                          DATASET_PATH + "/1/test",
                                                          srinivasan_classes,img_transformation= None)
srinivasan_dataset = torch.utils.data.ConcatDataset([srinivasan_dataset_train, srinivasan_dataset_val,
                                                     srinivasan_dataset_test])

# Calculate the average histograms
hist_group1 = avg_hist(kermany_dataset)
hist_group2 = avg_hist(srinivasan_dataset)
hist_group3 = avg_hist(oct500_dataset)

# Configure plot font and size
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14  # Adjust as needed

# Plot histograms
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(hist_group1, color='red', label='Kermany Data set')
ax.plot(hist_group2, color='blue', label='Srinivasan Data set')
ax.plot(hist_group3, color='green', label='OCT-500 Data Set')

ax.set_title('Average Normalized Histograms of three Data sets', fontsize=18) # Adjust as needed
ax.set_xlabel('Pixel Value', fontsize=14) # Adjust as needed
ax.set_ylabel('Normalized Count', fontsize=14) # Adjust as needed
ax.legend()

plt.tight_layout()
plt.show()