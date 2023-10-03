# Federated-Learning-for-Diagnosis-of-Age-Related-Macular-Degeneration

# Introduction

In our research, we delve into the application of federated learning (FL) for the classification of age-related macular degeneration (AMD) using optical coherence tomography image data. Utilizing both residual network and vision transformer encoders, our focus is on the binary classification of normal vs. AMD. Given the challenges posed by heterogeneous data distribution across different institutions, we've integrated four distinct domain adaptation techniques to mitigate domain shift issues.

Our findings underscore the potential of FL strategies, highlighting their ability to rival the performance of centralized models, even when each local model only accesses a fragment of the training data. Of particular note is the Adaptive Personalization FL strategy, which consistently showcased superior performance in our evaluations. This research not only sheds light on the effectiveness of simpler architectures in image classification tasks but also emphasizes the importance of data privacy and decentralization. We believe that our work paves the way for future investigations into more intricate models and diverse FL strategies, aiming for a deeper comprehension of their performance nuances.

## Methodology

We employ various federated learning methodologies in our research. The following are the methods used along with their respective references:
- **FedAvg**: [Link to Paper](https://arxiv.org/pdf/1602.05629.pdf?__s=xxxxxxx)
- **FedProx**: [Link to Paper](https://arxiv.org/pdf/1812.06127.pdf)
- **FedSR**: [Link to Paper](https://atuannguyen.com/assets/pdf/NeurIPS_nguyen2022fedsr.pdf)
- **FedMRI**: [Link to Paper](https://arxiv.org/pdf/2112.05752.pdf)
- **APFL**: [Link to Paper](https://arxiv.org/pdf/2003.13461.pdf)


# Dataset Repository

This repository contains datasets for various research purposes. The datasets can be accessed via the following link:

- [Link to DS1](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- [Link to DS2](https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm)
- [Link to DS3](https://ieee-dataport.org/open-access/octa-500)
## Dataset Structure

The dataset is organized into three main folders, each representing a different dataset:

```
dataset
    ├── 0:
    │   ├── train
    │   │   ├── AMD
    │   │   └── NORMAL
    │   └── test
    │       ├── AMD
    │       └── NORMAL
    ├── 1:
    │   ├── train
    │   │   ├── AMD1
    │   │   ├── ...
    │   │   ├── AMD12
    │   │   ├── NORMAL1
    │   │   ├── ...
    │   │   └── NORMAL12
    │   └── test
    │       ├── AMD13
    │       ├── ...
    │       ├── AMD15
    │       ├── NORMAL13
    │       ├── ...
    │       └── NORMAL15
    └── 2:
        ├── train
        │   ├── OCTA_3mm
        │   └── OCTA_6mm
        └── test
            ├── OCTA_3mm
            └── OCTA_6mm
```

## Configuration
There is an .env file located in ./data/.env. Ensure to set the dataset path in this file before proceeding with any operations.
Certainly! Here's a suggested addition to your GitHub README file for the installation part using the `environment.yml` file:

---

## Installation

To set up the environment and dependencies required for this project, we provide an `environment.yml` file. Follow the steps below to create a Conda virtual environment using this file:

1. **Clone the Repository**:
   ```bash
   git clone [YOUR_GITHUB_REPO_LINK]
   cd [YOUR_REPO_NAME]
   ```

2. **Install Conda**:
   If you haven't installed Conda yet, download and install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

3. **Create a Conda Environment**:
   Use the `environment.yml` file to create a new Conda environment:
   ```bash
   conda env create -f environment.yml
   ```

4. **Activate the Environment**:
   ```bash
   conda activate [YOUR_ENVIRONMENT_NAME]
   ```

   Note: Replace `[YOUR_ENVIRONMENT_NAME]` with the name of the environment specified in the `environment.yml` file. If you didn't specify a name, it defaults to the directory name.

5. **Run the Code**:
   Now that you have activated the environment, you can run the code in this repository.

---

Make sure to replace `[YOUR_GITHUB_REPO_LINK]` with the actual link to your GitHub repository and `[YOUR_REPO_NAME]` with the name of your repository. If you've specified a custom name for the environment in the `environment.yml` file, replace `[YOUR_ENVIRONMENT_NAME]` with that name. Otherwise, users can use the directory name as the environment name.

