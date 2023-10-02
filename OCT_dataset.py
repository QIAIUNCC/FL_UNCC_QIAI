import math
import random
import torch
import warnings
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from torch.utils.data import Dataset
import os
from PIL import Image
from natsort import natsorted
import subsetsum as sb

IMAGE_SIZE = (512, 496)


class OCTDataset(Dataset):

    def __init__(self, data_root, img_type="L", transform=None, img_size=IMAGE_SIZE,
                 dataset_func=None, nst_prob=1, **kwargs):

        self.data_root = data_root  # root directory
        self.transform = transform  # transform functions
        self.img_size = img_size  # the image size
        self.img_type = img_type  # the type of image L, R
        self.is_oct_500 = False  # whether it is oct-500 dataset or not
        self.style_hdf5_path = None  # path to the nst dataset
        self.classes = kwargs["classes"]  # list of tuple of class names and labels: [(NORMAL,0), ...]
        if "oct500" in kwargs:
            self.is_oct_500 = True
        if "img_paths" in kwargs:
            self.imgPaths_labels = kermany_split_data(kwargs["img_paths"], val_split=kwargs["val_split"],
                                                     mode=kwargs["mode"],
                                                     classes=kwargs["classes"])
        else:
            self.imgPaths_labels = dataset_func(self.data_root, **kwargs)

        self.nst_prob = nst_prob

    def __getitem__(self, index):
        img_path, label = self.imgPaths_labels[index]
        img_path = img_path.replace("\\", "/")

        if random.uniform(0, 1) < self.nst_prob and self.style_hdf5_path is not None:
            img = self.load_nst_img(img_path)
        else:
            image = self.load_img(img_path)
            if self.transform is not None:
                img = self.transform(image)
            else:
                # throws error
                img = image
            # image.show()

        results = dict(img_path=img_path, img_name=img_path.split(self.data_root)[1].split("/")[1], img=img,
                       label=label)
        return results

    def __len__(self):
        """
        The __len__ should be overridden when the parent is Dataset to return the number of elements of the dataset

        Return:
            - returns the size of the dataset
        """
        return len(self.imgPaths_labels)

    def load_img(self, img_path):
        img = Image.open(img_path).convert(self.img_type)
        return img

    def split(self, val_split=0.1):
        train_dataset = OCTDataset(data_root=self.data_root, img_type=self.img_type, transform=self.transform,
                                   img_size=IMAGE_SIZE,
                                   style_hdf5_path=self.style_hdf5_path, mode="train", val_split=val_split,
                                   img_paths=self.imgPaths_labels)

        val_dataset = OCTDataset(data_root=self.data_root, img_type=self.img_type, transform=self.transform,
                                 img_size=IMAGE_SIZE, classes=self.classes,
                                 style_hdf5_path=self.style_hdf5_path, mode="val", val_split=val_split,
                                 img_paths=self.imgPaths_labels)

        return train_dataset, val_dataset


def get_srinivasan_imgs(data_root: str, **kwargs):
    """

    :param data_root:
    :param kwargs:
        - ignore_folders (np.array): indices of files to ignore
        - sub_folders_name (str): path containing the subfolders
        - classes (list of tuples): ex: [("NORMAL", 0), ...]
    :return:
    """
    classes = kwargs["classes"]
    img_filename_list = []
    for c in classes:
        img_filename_list += list(filter(lambda k: c[0] in k, os.listdir(os.path.join(data_root))))

    imgs_path = []
    for img_file in img_filename_list:
        if any(item == int(img_file.replace("AMD", "").replace("NORMAL", "").replace("DME", ""))
               for item in kwargs["ignore_folders"]):
            continue
        folder = os.path.join(data_root, img_file, kwargs["sub_folders_name"])
        imgs_path += [(os.path.join(folder, id), get_class(os.path.join(folder, id), kwargs["classes"]))
                      for id in os.listdir(folder)]
     
    return imgs_path


def get_kermany_imgs(data_root: str, **kwargs):
    classes = kwargs["classes"]
    img_paths = []
    img_filename_list = []
    path = os.listdir(os.path.join(data_root))
    for c in classes:
        img_filename_list += list(filter(lambda k: c[0] in k, path))

    for img_file in img_filename_list:
        img_file_path = os.path.join(data_root, img_file)
        # sort lexicographically the img names in the img_file directory
        img_names = natsorted(os.listdir(img_file_path))
        # split the first part of image
        # dictionary{patient_id:[list of visits]}
        img_dict = {}
        for img in img_names:
            img_count = img.split("-")[2]
            img_name = img.replace(img_count, "")
            if img_name in img_dict:
                img_dict[img_name] += [img_count]
            else:
                img_dict[img_name] = [img_count]
        if "cv" in kwargs:
            assert kwargs["cv"] > 1
            cv_len = len(img_dict.keys()) // kwargs["cv"]
            start_idx = kwargs["cv_counter"] * cv_len
            end_idx = start_idx + cv_len if kwargs["cv_counter"] < kwargs["cv"] - 1 else len(img_dict.keys())

            keys = list(img_dict.keys())
            for key, val in img_dict.items():
                if key not in keys[start_idx:end_idx] and kwargs["mode"] == "train":
                    img_paths += [(img_file_path + "/" + key + count, get_class(img_file_path + key, classes)) for count
                                  in val]
                elif key in keys[start_idx:end_idx] and kwargs["mode"] == "val":
                    img_paths += [(img_file_path + "/" + key + count, get_class(img_file_path + key, classes)) for count
                                  in val]
        elif "val_split" in kwargs:
            # create a list of #visits of all clients
            num_visits = [len(n) for n in img_dict.values()]
            total_imgs = sum(num_visits)
            val_num = math.floor(total_imgs * kwargs["val_split"])
            for solution in sb.solutions(num_visits, val_num):
                # `solution` contains indices of elements in `nums`
                subset = [i for i in solution]
                break
            keys = list(img_dict.keys())
            temp_paths = []
            for idx in subset:
                if kwargs["mode"] == "val":
                    temp_paths += [(img_file_path + "/" + keys[idx] + count,
                                    get_class(img_file_path + keys[idx], classes)) for count in img_dict[keys[idx]]]
            if "limit_val" in kwargs and kwargs["limit_val"] > 0:
                temp_paths = temp_paths[0:kwargs["limit_val"] // len(classes)]

            counter = -1
            for key, val in img_dict.items():
                counter += 1
                if counter in subset:
                    continue
                elif kwargs["mode"] == "train":
                    temp_paths += [(img_file_path + "/" + key + count, get_class(img_file_path + key, classes))
                                   for count in val]
            if "limit_train" in kwargs and kwargs["limit_train"] > 0:
                temp_paths = temp_paths[0:kwargs["limit_train"] // len(classes)]
            img_paths += temp_paths

        else:  # in case of test mode or when no cv or split is specified
            temp_paths = []
            for key, val in img_dict.items():
                temp_paths += [(img_file_path + "/" + key + count, get_class(img_file_path + key, classes))
                               for count in val]
            if "limit_test" in kwargs and kwargs["limit_test"] > 0:
                temp_paths = temp_paths[0:kwargs["limit_test"] // len(classes)]
            img_paths += temp_paths
    return img_paths


def kermany_split_data(img_paths, val_split, mode, classes):
    img_paths_out = []
    for c in classes:
        filtered = list(filter(lambda k: c[0] in k, img_paths))
        img_dict = {}
        for img in filtered:
            img_count = img.split("-")[2]
            img_name = img.replace(img_count, "")
            if img_name in img_dict:
                img_dict[img_name].append(img_count)
            else:
                img_dict[img_name] = [img_count]

        num_visits = [len(n) for n in img_dict.values()]
        total_imgs = sum(num_visits)
        val_num = math.floor(total_imgs * val_split)
        for solution in sb.solutions(num_visits, val_num):
            # `solution` contains indices of elements in `nums`
            subset = [i for i in solution]
            break
        keys = list(img_dict.keys())
        counter = -1
        for idx in subset:
            if mode == "val":
                img_paths_out += [(keys[idx] + count, get_class(keys[idx], classes)) for count in img_dict[keys[idx]]]

        for key, val in img_dict.items():
            counter += 1
            if counter in subset:
                continue
            if mode == "train":
                img_paths_out += [(key + count, get_class(key, classes)) for count in val]
        print(mode, c[0], len(img_paths_out))
    return img_paths_out


def get_class(img_name, classes: list):
    for c, v in classes:
        if c in img_name:
            return v


def get_oct500_imgs(data_root: str, **kwargs):
    assert sum(kwargs["train_val_test"]) == 1
    classes = kwargs["classes"]
    mode = kwargs["mode"]
    train_val_test = kwargs["train_val_test"]

    df = pd.read_excel(os.path.join(data_root, "Text labels.xlsx"))
    img_paths = []
    for c in classes:
        temp_path = []
        disease_ids = df[df["Disease"] == c[0]]["ID"].sort_values().tolist()
        train, val, test = split_oct500(disease_ids, train_val_test)
        # print("len train",c[0], len(train))
        # print("len val"c[0], len(val))
        # print("len test"c[0], len(test))
        if mode == "train":
            temp_path += get_oct500(train, data_root, class_label=c[1])
            if "limit_train" in kwargs and kwargs["limit_train"] > 0:
                temp_path = temp_path[0:kwargs["limit_train"] // len(classes)]
        elif mode == "val":
            temp_path += get_oct500(val, data_root, class_label=c[1])
            if "limit_val" in kwargs and kwargs["limit_val"] > 0:
                temp_path = temp_path[0:kwargs["limit_val"] // len(classes)]
        elif mode == "test":
            temp_path += get_oct500(test, data_root, class_label=c[1])
            if "limit_test" in kwargs and kwargs["limit_test"] > 0:
                temp_path = temp_path[0:kwargs["limit_test"] // len(classes)]
        img_paths += temp_path
    return img_paths
    # train_amd, val_amd, test_amd = split_oct500(amd_ids, train_val_test)
    # train_dr, val_dr, test_dr = split_oct500(dr_ids, train_val_test)

    #
    # print("len val normal", len(val_normal))
    # print("len val amd", len(val_amd))
    # print("len val dr", len(val_dr))
    #
    # print("len test normal", len(test_normal))
    # print("len test amd", len(test_amd))
    # print("len test dr", len(test_dr))


def split_oct500(total_ids: list, train_val_test: tuple):
    """
    Divides data into train, val, test
    :param total_ids: list of patients ids
    :param train_val_test: (train split, val split, test split) --> the sum should be 1
    """
    train_idx = math.ceil(len(total_ids) * train_val_test[0])
    return total_ids[0: train_idx], \
        total_ids[train_idx + 1: math.ceil(len(total_ids) * train_val_test[1]) + train_idx], \
        total_ids[math.ceil(len(total_ids) * train_val_test[1]) + train_idx + 1:]


def get_oct500(list_ids, data_root, class_label):
    img_paths = []
    for idd in list_ids:
        file_path = os.path.join(data_root, "OCT", str(idd))
        for img in os.listdir(file_path):
            if ("6mm" in data_root and 160 <= int(img[:-4]) <= 240) or \
                    ("3mm" in data_root and 100 <= int(img[:-4]) <= 180):
                img_paths.append((os.path.join(file_path, img), class_label))
        # img_paths += [(os.path.join(file_path, img), class_label)
        #               for img in os.listdir(file_path)]
    return img_paths


def get_dr_imgs(data_root: str, **kwargs):
    img_folders = os.listdir(os.path.join(data_root))
    imgs_dict = {}
    img_ids = []
    for img_folder in img_folders:
        imgs = natsorted(os.listdir(os.path.join(data_root, img_folder)))
        for img in imgs:
            img_path = os.path.join(data_root, img_folder, img)
            patient_name = img.split("_")[1] + img.split("_")[2]
            if patient_name in imgs_dict:
                imgs_dict[patient_name] += [img_path]
            else:
                imgs_dict[patient_name] = [img_path]
        if kwargs["mode"] == "test":
            for key, val in imgs_dict.items():
                img_ids += [img_p for img_p in val]
        else:
            cv_len = len(imgs_dict.keys()) // kwargs["cv"]
            start_idx = kwargs["cv_counter"] * cv_len
            end_idx = start_idx + cv_len if kwargs["cv_counter"] < kwargs["cv"] - 1 else len(imgs_dict.keys())
            keys = list(imgs_dict.keys())
            for key, val in imgs_dict.items():
                if key not in keys[start_idx:end_idx] and kwargs["mode"] == "train":
                    img_ids += [img_p for img_p in val]
                elif key in keys[start_idx:end_idx] and kwargs["mode"] == "val":
                    img_ids += [img_p for img_p in val]
    return img_ids


if __name__ == "__main__":
    # read from the .env file
    load_dotenv(dotenv_path="./data/.env")
    DATASET_PATH = os.getenv('DATASET_PATH')
    OCTDataset(data_root=DATASET_PATH + "2/OCTA_6mm/", dataset_func=get_oct500_imgs,
               mode="train", train_val_test=(0.6, 0.2, 0.2))
