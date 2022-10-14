from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile
import logging
import glob

from .bases import ImageDataset
from . import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Canifa(ImageDataset):
    """
    only train data

    id1
        image.jpg
    id2
        image.jpg
    """
    dataset_dir = 'canifa_sup'
    dataset_name = "canifa"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        train = self._process_dir(self.dataset_dir)
        query = []
        gallery = []

        super(Canifa, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, is_train=True):
        cls_paths = list(glob.glob(f"{dir_path}/*"))

        dataset = []
        cam_id = {}
        for pid, cls_path in enumerate(cls_paths):
            for img_path in list(glob.glob(f"{cls_path}/*.jpg")):
                cam_id = int(osp.basename(img_path).split("_")[0])

                if is_train:
                    dataset.append((
                        img_path,
                        self.dataset_name + "_" + str(pid),
                        self.dataset_name + "_" + str(cam_id)
                    ))
                else:
                    dataset.append((img_path, pid, cam_id))
                

        return dataset


@DATASET_REGISTRY.register()
class Tokyolife(ImageDataset):
    """
    only train data

    id1
        image.jpg
    id2
        image.jpg
    """
    dataset_dir = 'tokyolife_sup'
    dataset_name = "tokyolife"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        train = self._process_dir(self.dataset_dir)
        query = []
        gallery = []

        super(Tokyolife, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, is_train=True):
        cls_paths = list(glob.glob(f"{dir_path}/*"))

        dataset = []
        cam_id = {}
        for pid, cls_path in enumerate(cls_paths):
            for img_path in list(glob.glob(f"{cls_path}/*.jpg")):
                cam_id = int(osp.basename(img_path).split("_")[0])

                if is_train:
                    dataset.append((
                        img_path,
                        self.dataset_name + "_" + str(pid),
                        self.dataset_name + "_" + str(cam_id)
                    ))
                else:
                    dataset.append((img_path, pid, cam_id))
                

        return dataset


@DATASET_REGISTRY.register()
class Genviet(ImageDataset):
    """
    only train data

    id1
        image.jpg
    id2
        image.jpg
    """
    dataset_dir = 'genviet_sup'
    dataset_name = "genviet"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        train = self._process_dir(self.dataset_dir)
        query = []
        gallery = []

        super(Genviet, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, is_train=True):
        cls_paths = list(glob.glob(f"{dir_path}/*"))

        dataset = []
        cam_id = {}
        for pid, cls_path in enumerate(cls_paths):
            for img_path in list(glob.glob(f"{cls_path}/*.jpg")):
                cam_id = int(osp.basename(img_path).split("_")[0])

                if is_train:
                    dataset.append((
                        img_path,
                        self.dataset_name + "_" + str(pid),
                        self.dataset_name + "_" + str(cam_id)
                    ))
                else:
                    dataset.append((img_path, pid, cam_id))
                

        return dataset
