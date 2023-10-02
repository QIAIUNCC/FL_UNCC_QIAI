import math
import os
import mmcv
import random
from PIL import Image
from natsort import natsorted

new_root = "/home/sgholami/Downloads/test"


def move_data(img_path):
    image = Image.open(img_path)
    img_path = img_path.replace("\\", "/")
    # p2 = os.path.join(root, img.split("-")[0], img.split("-")[0]+img.split(img.split("-")[2]))
    p2 = os.path.join(new_root, img_path.split("/")[-2], img_path.split("/")[-1])
    image.save(p2)
    os.remove(img_path)


if __name__ == "__main__":
    root = "/home/sgholami/Downloads/kermany_full_no_duplicates"
    # root = "/sina/flower/data/3dataset/0"
    image_format = ".jpeg"
    nproc = 8
    percentage = 0.05
    files = [
        # "CNV",
        # "DME",
        # "AMD",
        "DRUSEN",
        # "NORMAL"
    ]
    img_names = {}
    img_filename_list = os.listdir(os.path.join(root))
    filtered = list(filter(lambda k: k in files, img_filename_list))
    print(filtered)
    train_img_ids = []
    test_img_ids = []
    for img_file in filtered:
        img_file_path = os.path.join(root, img_file)
        # sort lexicographically the img names in the img_file directory
        img_names = natsorted(os.listdir(img_file_path))
        # split the first part of image
        imgs_dict = {}
        for img in img_names:
            # get the patient image counter
            img_count = img.split("-")[2]
            # get the patient id
            img_id = img.replace(img_count, "")
            # save the patient ids and patient counters in a dictionary
            if img_id in imgs_dict:
                imgs_dict[img_id] += [img_count]
            else:
                imgs_dict[img_id] = [img_count]
        val_data_len = math.floor(len(imgs_dict.keys()) * percentage)
        keys = list(imgs_dict.keys())
        for key, val in imgs_dict.items():
            # if key not in keys[:val_data_len]:
            #     train_img_ids += [img_file_path + "/" + key + count for count in val]
            #     print(train_img_ids)
            if key in keys[:val_data_len]:
                test_img_ids += [img_file_path + "/" + key + count for count in val]
    # images = os.listdir(os.path.join(root, files[0]))
    # images_idx = random.sample(range(0, len(images)), math.floor(len(images) * percentage))
    # tasks = []
    # for i in images_idx:
    #     img_name = images[i].split(image_format)[0]
    #     tasks.append({"root": root,
    #                   "files": files,
    #                   "img_name": img_name,
    #                   "image_format": image_format,
    #                   "out_suffix": out_suffix})
    #
    if nproc > 1:  # handle it in parallel or not
        mmcv.track_parallel_progress(move_data, test_img_ids, nproc)
    else:
        mmcv.track_progress(move_data, test_img_ids)
