"""
Find the difference between two folders
To check whether the segmentation (classification) and image folders have the same images
"""
import os
import shutil

import cv2
import natsort
import numpy as np
from natsort import natsorted


def diff(first_dir, second_dir, first_dir_img_format=".png", second_dir_img_format=".png", remove_diff=False):
    """
    Finds the images that common in both first_dir and second_dir based on their names disregarding their format.
     It is possible to remove those images that are ont inside their intersection.

    Args:
        -first_dir (str): path to the first directory
        - first_dir_img_format (str): the format of the images inside the first_dir
        -second_dir (str): path to the second directory
        - second_dir_img_format (str): the format of the images inside the second_dir
        -remove_diff (bool): whether to remove images that are not in their intersection
    """
    d1 = sorted(os.listdir(first_dir))
    d2 = sorted(os.listdir(second_dir))
    f1 = []
    for f in d1:
        f1.append(f.split(first_dir_img_format)[0])

    f2 = []
    for f in d2:
        f2.append(f.split(second_dir_img_format)[0])

    intersection = list(set(f2).intersection(set(f1)))
    print("intersection between two files", len(intersection))

    if remove_diff:
        for f in intersection:
            if os.stat(os.path.join(first_dir, f) + first_dir_img_format).st_size == \
                    os.stat(os.path.join(second_dir, f) + second_dir_img_format).st_size:
                os.remove(os.path.join(second_dir, f) + second_dir_img_format)

        # for f in intersection:
        #     os.remove(os.path.join(first_dir, f) + first_dir_img_format)


def rename_to_new(train_dir, test_dir, first_dir_img_format=".png", test_img_format=".png", ):
    img_names_train = natsorted(os.listdir(train_dir))
    img_names_test = natsorted(os.listdir(test_dir))
    img_dict = {}
    for img in img_names_train:
        img_count = img.split("-")[2]
        img_name = img.replace(img_count, "")
        if img_name in img_dict:
            img_dict[img_name] += [img_count]
        else:
            img_dict[img_name] = [img_count]

    for img in img_names_test:
        img_count = img.split("-")[2]
        img_name = img.replace(img_count, "")
        if img_name in img_dict:
            print(test_dir + "/" + img)
            print(img_dict[img_name][-1])
            new_count = int(img_dict[img_name][-1].replace(test_img_format, "")) + 1
            os.rename(test_dir + "/" + img, test_dir + "/" + img_name + str(new_count) + test_img_format)
            img_dict[img_name] += [str(new_count) + test_img_format]


# def rename_to_new(first_dir, second_dir, first_dir_img_format, second_dir_img_format):
#     d1 = sorted(os.listdir(first_dir))
#     d2 = sorted(os.listdir(second_dir))
#     f1 = []
#     for f in d1:
#         f1.append(f.split(first_dir_img_format)[0])
#
#     f2 = []
#     for f in d2:
#         f2.append(f.split(second_dir_img_format)[0])
#
#     intersection = list(set(f2).intersection(set(f1)))
#     for img_name in intersection:
def mse(img1, img2):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    if h1 != h2 or w1 != w2:
        return 100
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff ** 2)
    mse = err / (float(h1 * w1))
    return mse


def check_duplicate(train_dir):
    img_names_train = natsorted(os.listdir(train_dir))
    img_dict = {}
    for img in img_names_train:
        img_count = img.split("-")[2]
        img_name = img.replace(img_count, "")
        if img_name in img_dict:
            img_dict[img_name] += [img]
        else:
            img_dict[img_name] = [img]

    duplicate = {}
    for i in img_dict:
        print("next")
        for j in range(0, len(img_dict[i]) - 1):
            img1 = cv2.imread(os.path.join(train_dir, img_dict[i][j]), cv2.IMREAD_GRAYSCALE)
            for z in range(j + 1, len(img_dict[i])):
                img2 = cv2.imread(os.path.join(train_dir, img_dict[i][z]), cv2.IMREAD_GRAYSCALE)
                if mse(img1, img2) == 0:
                    print(img_dict[i][j], img_dict[i][z])
                    if img_dict[i][z] in duplicate:
                        duplicate[img_dict[i][z]] += [img_dict[i][j]]
                    else:
                        duplicate[img_dict[i][z]] = [img_dict[i][j]]

                    # if z not in duplicate:
                    #     duplicate.append(z)
                    # if j not in duplicate:
                    #     duplicate.append(j)
    for dup in duplicate:
        for img in duplicate[dup]:
            try:
                os.remove(os.path.join(train_dir, img))
            except:
                print("deleted before!")


def check_duplicate_all(train_dir):
    folders = os.listdir(train_dir)
    imgs = []
    for f in folders:
        imgs += os.listdir(os.path.join(train_dir, f))
    for i in range(0, len(imgs) - 1):
        img1 = cv2.imread(os.path.join(train_dir, imgs[i].split("-")[0], imgs[i]), cv2.IMREAD_GRAYSCALE)
        for j in range(i + 1, len(imgs)):
            img2 = cv2.imread(os.path.join(train_dir, imgs[j].split("-")[0], imgs[j]), cv2.IMREAD_GRAYSCALE)
            if mse(img1, img2) == 0:
                print(imgs[i], imgs[j])


if __name__ == "__main__":
    check_duplicate_all(train_dir="/home/sgholami/Downloads/OCT2017")
    # check_duplicate(train_dir="/home/sgholami/Downloads/OCT2017/CNV")
    # check_duplicate(train_dir="/home/sgholami/Downloads/OCT2017/NORMAL")
    # check_duplicate(train_dir="/home/sgholami/Downloads/OCT2017/DRUSEN")
    # check_duplicate(train_dir="/home/sgholami/Downloads/OCT2017/DME")

    # rename_to_new(train_dir="/home/sgholami/Downloads/OCT2017/train/NORMAL",
    #               test_dir="/home/sgholami/Downloads/OCT2017/test/NORMAL",
    #               first_dir_img_format=".jpeg",
    #               test_img_format=".jpeg")
    #
    # rename_to_new(train_dir="/home/sgholami/Downloads/OCT2017/train/CNV",
    #               test_dir="/home/sgholami/Downloads/OCT2017/test/CNV",
    #               first_dir_img_format=".jpeg",
    #               test_img_format=".jpeg")
    #
    # rename_to_new(train_dir="/home/sgholami/Downloads/OCT2017/train/DME",
    #               test_dir="/home/sgholami/Downloads/OCT2017/test/DME",
    #               first_dir_img_format=".jpeg",
    #               test_img_format=".jpeg")
    # #
    # rename_to_new(train_dir="/home/sgholami/Downloads/OCT2017/train/DRUSEN",
    #               test_dir="/home/sgholami/Downloads/OCT2017/test/DRUSEN",
    #               first_dir_img_format=".jpeg",
    #               test_img_format=".jpeg")
    #
    # copy_to_new(train_dir="/home/sgholami/Downloads/OCT2017/train/DME",
    #             test_dir="/home/sgholami/Downloads/OCT2017/test/DME",
    #             first_dir_img_format=".jpeg",
    #             test_img_format=".jpeg",
    #             save_dir="/home/sgholami/Downloads/OCT2017/similar/DME")
    #
    # copy_to_new(train_dir="/home/sgholami/Downloads/OCT2017/train/DRUSEN",
    #             test_dir="/home/sgholami/Downloads/OCT2017/test/DRUSEN",
    #             first_dir_img_format=".jpeg",
    #             test_img_format=".jpeg",
    #             save_dir="/home/sgholami/Downloads/OCT2017/similar/DRUSEN")
    # rename_to_new(first_dir="/home/sgholami/Downloads/OCT2017/similar/NORMAL",
    #             second_dir="/home/sgholami/Downloads/OCT2017/train/NORMAL",
    #             first_dir_img_format=".jpeg",
    #             second_dir_img_format=".jpeg")

    # diff(first_dir="/home/sgholami/Downloads/OCT2017/train/NORMAL",
    #      second_dir="/home/sgholami/Downloads/OCT2017/similar/NORMAL",
    #      first_dir_img_format=".jpeg",
    #      second_dir_img_format=".jpeg",
    #      remove_diff=True)
    #
    # diff(first_dir="/home/sgholami/Downloads/OCT2017/train/CNV",
    #      second_dir="/home/sgholami/Downloads/OCT2017/similar/CNV",
    #      first_dir_img_format=".jpeg",
    #      second_dir_img_format=".jpeg",
    #      remove_diff=True)
    #
    # diff(first_dir="/home/sgholami/Downloads/OCT2017/train/DRUSEN",
    #      second_dir="/home/sgholami/Downloads/OCT2017/similar/DRUSEN",
    #      first_dir_img_format=".jpeg",
    #      second_dir_img_format=".jpeg",
    #      remove_diff=True)
    #
    # diff(first_dir="/home/sgholami/Downloads/OCT2017/train/DME",
    #      second_dir="/home/sgholami/Downloads/OCT2017/similar/DME",
    #      first_dir_img_format=".jpeg",
    #      second_dir_img_format=".jpeg",
    #      remove_diff=True)
