import json
import os
import re

import numpy as np
import pandas as pd
import scipy

if __name__ == "__main__":
    metrics = ['test_precision_NORMAL', 'test_accuracy_AMD', 'test_precision_AMD', 'f1_score', 'auc', 'loss']
    log_root = "./FedAvg/ViT/log_e10"
    rep = 10

    res_dic = {}
    res_avg = {}
    res_max = {}
    res_min = {}
    res_std = {}
    for f in os.listdir(log_root):
        if "server.txt" in f or "val" in f:
            continue
        res_dic[f] = []
        with open(os.path.join(log_root, f), "r") as ff:
            lines = ff.readlines()
            for i, l in enumerate(lines):
                if "==" in l:
                    num_list = int(re.findall(r'\d+', l)[0])
                    if i + 2 >= len(lines):
                        res_dic[f].append(json.loads(lines[i + 1].replace("\'", "\"")))
                    else:
                        num_list2 = int(re.findall(r'\d+', lines[i + 2])[0])
                        if num_list2 > num_list:
                            continue
                        else:
                            # print(lines[i + 1])
                            res_dic[f].append(json.loads(lines[i + 1].replace("\'", "\"")))
        df = pd.DataFrame(res_dic[f])
        # print(df.auc)
        dataset = f.replace(')', '').replace('(', '').replace(',', '').split('_')[0]
        print(f"========={dataset}=========")
        # print("=========mean=========")
        # print(df.mean())
        # print("=========std=========")
        # print(df.std())
        sem = scipy.stats.sem(df)
        coef = 1.96
        start = df.mean() - coef * sem
        end = df.mean() + coef * sem
        for metric in metrics:
            values = df[metric]

            # Compute the mean
            mean_val = np.mean(values)
            # Compute the standard error of the mean (SEM)
            sem = scipy.stats.sem(values)
            # Define your confidence interval
            confidence = 0.95
            h = sem * scipy.stats.t.ppf((1 + confidence) / 2., len(values) - 1)

            print(f'{metric}: {str(float(f"{mean_val*100:.2f}"))} ± {str(float(f"{h*100:.2f}"))}')


       
