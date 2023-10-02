import json
import os

import numpy as np
import pandas as pd
import scipy

if __name__ == "__main__":
    metrics = ['test_precision_NORMAL', 'test_accuracy_AMD', 'test_precision_AMD', 'f1_score', 'auc', 'loss']

    log_root = "./log_e10/"
    rep = 10
    for f in os.listdir(log_root):
        if "PREV" in f:
            continue

        res_dic = []
        with open(os.path.join(log_root, f), "r") as ff:
            lines = ff.readlines()
            for i, l in enumerate(lines):
                if "==" not in l:
                    res_dic.append(json.loads(lines[i].replace("\'", "\"")))
        df = pd.DataFrame(res_dic)
        print(f"========={f}=========")
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

            print(f'{metric}: {str(float(f"{mean_val * 100:.2f}"))} ± {str(float(f"{h * 100:.2f}"))}')

