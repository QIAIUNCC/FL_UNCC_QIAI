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
.
├── DS1: Kermany et al. dataset
│   ├── training
│   └── test
├── DS2: Serinivasan et al. dataset
│   ├── training
│   └── test
└── DS3: OCT-500 dataset
    ├── training
    └── test
```

## Configuration
There is an .env file located in ./data/.env. Ensure to set the dataset path in this file before proceeding with any operations.

