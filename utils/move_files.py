import os
import shutil

path_dir = "/sina/contrastive/data/kagggg/duplicates"
dest_dir = "/sina/contrastive/data/kagggg/"
folders = os.listdir(path_dir)

for f in folders:
    if "NORMAL" not in f:
        imgs = os.listdir(os.path.join(path_dir, f))
        if len(imgs) == 0:
            os.rmdir(os.path.join(path_dir, f))
        for img in imgs:
            shutil.copy(os.path.join(path_dir, f, img), os.path.join(dest_dir, img.split("-")[0], img))
            os.remove(os.path.join(path_dir, f, img))
