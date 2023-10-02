import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
# Load the font
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()


def format_float(num):
    return f"{num:.4f}".rstrip('0').rstrip('.')


def plot_auc(model, test_datasets, log_name, title):
    csfont = {'fontname': 'Times New Roman'}
    # Suppose you have predictions and targets
    # Plot
    plt.figure()
    lw = 2
    colors = ["red", "blue", "green"]
    # Collect all predictions and targets
    i = 0
    model.eval()
    with torch.no_grad():
        for key, val in test_datasets.items():
            targets = []
            outputs = []
            for batch in val:
                x, y = batch["img"], batch["label"]
                output = model.forward(x.to(model.device))

                if isinstance(output, tuple):
                    preds = output[0].argmax(-1)
                else:
                    preds = output.argmax(-1)

                targets.extend(y.detach().cpu().numpy())
                outputs.extend(preds.detach().cpu().numpy())

            fpr, tpr, _ = roc_curve(targets, outputs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr,
                     label=f'ROC curve over {key} (area = {format_float(roc_auc)})',
                     linestyle='--',
                     color=colors[i], linewidth=2)
            i += 1

    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', **csfont)
    plt.ylabel('True Positive Rate', **csfont)
    plt.title(f'ROC of {title} Model', **csfont)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(log_name + ".png")
