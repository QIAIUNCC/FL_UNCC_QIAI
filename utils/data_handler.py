from torch.utils.data import DataLoader

from OCT_dataset import OCTDataset, get_oct500_imgs, get_srinivasan_imgs, get_kermany_imgs


def get_data_loaders(train_dataset, val_dataset, batch_size, num_workers=4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              drop_last=False, pin_memory=True, num_workers=num_workers)
    return train_loader, valid_loader


def get_oct500_datasets(dataset_path, classes, img_transformation, limit_train=0, limit_val=0, limit_test=0):
    train_val_test = (0.85, 0.025, 0.125)
    oct500_dataset_train = OCTDataset(data_root=dataset_path,
                                      transform=img_transformation,
                                      classes=classes,
                                      mode="train",
                                      train_val_test=train_val_test,
                                      dataset_func=get_oct500_imgs,
                                      limit_train=limit_train
                                      )
    oct500_dataset_val = OCTDataset(data_root=dataset_path,
                                    transform=img_transformation,
                                    classes=classes,
                                    mode="val",
                                    train_val_test=train_val_test,
                                    dataset_func=get_oct500_imgs,
                                    limit_val=limit_val
                                    )

    oct500_dataset_test = OCTDataset(data_root=dataset_path,
                                     transform=img_transformation,
                                     classes=classes,
                                     mode="test",
                                     train_val_test=train_val_test,
                                     dataset_func=get_oct500_imgs,
                                     limit_test=limit_test
                                     )
    return oct500_dataset_train, oct500_dataset_val, oct500_dataset_test


def get_srinivasan_datasets(train_path, test_path, classes, img_transformation):
    srinivasan_dataset_train = OCTDataset(data_root=train_path,
                                          transform=img_transformation,
                                          classes=classes,
                                          ignore_folders=[12, 13, 14, 15],
                                          sub_folders_name="TIFFs/8bitTIFFs",
                                          dataset_func=get_srinivasan_imgs
                                          )
    srinivasan_dataset_val = OCTDataset(data_root=train_path,
                                        transform=img_transformation,
                                        classes=classes,
                                        ignore_folders=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15],
                                        sub_folders_name="TIFFs/8bitTIFFs",
                                        dataset_func=get_srinivasan_imgs
                                        )

    srinivasan_dataset_test = OCTDataset(data_root=test_path,
                                         transform=img_transformation,
                                         classes=classes,
                                         ignore_folders=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                         sub_folders_name="TIFFs/8bitTIFFs",
                                         dataset_func=get_srinivasan_imgs
                                         )
    return srinivasan_dataset_train, srinivasan_dataset_val, srinivasan_dataset_test


def get_kermany_datasets(train_path, test_path, classes, img_transformation, limit_train=-1, limit_val=-1,
                         limit_test=-1, val_split=0.2):
    kermany_dataset_train = OCTDataset(data_root=train_path,
                                       transform=img_transformation,
                                       classes=classes,
                                       mode="train",
                                       val_split=val_split,
                                       dataset_func=get_kermany_imgs,
                                       limit_train=limit_train
                                       )
    kermany_dataset_val = OCTDataset(data_root=train_path,
                                     transform=img_transformation,
                                     classes=classes,
                                     mode="val",
                                     val_split=val_split,
                                     dataset_func=get_kermany_imgs,
                                     limit_val=limit_val
                                     )

    kermany_dataset_test = OCTDataset(data_root=test_path,
                                      transform=img_transformation,
                                      classes=classes,
                                      mode="test",
                                      dataset_func=get_kermany_imgs,
                                      limit_test=limit_test
                                      )
    return kermany_dataset_train, kermany_dataset_val, kermany_dataset_test


def get_datasets_classes():
    srinivasan_classes = [("NORMAL", 0),
                          ("AMD", 1),
                          ]

    kermany_classes = [("NORMAL", 0),
                       ("AMD", 1),
                       ]

    oct500_classes = [("NORMAL", 0),
                      ("AMD", 1),
                      ]
    return srinivasan_classes, kermany_classes, oct500_classes
