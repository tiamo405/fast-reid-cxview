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


@DATASET_REGISTRY.register()
class OccludedREID(ImageDataset):
    """
    only test data

    id1
        image.jpg
    id2
        image.jpg
    """
    dataset_dir = 'occludereid'
    dataset_name = "OccludedREID"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir, self.dataset_name)

        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        required_files = [
            self.dataset_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = []
        query = self._process_dir(self.query_dir, is_train=False)
        gallery = self._process_dir(self.gallery_dir, is_train=False)

        super(OccludedREID, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_(\d*)')

        dataset = []
        for img_path in img_paths:
            pid, cam_id = map(int, pattern.search(img_path).groups())
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
class PartialREID(ImageDataset):
    """
    only test data

    id1
        image.jpg
    id2
        image.jpg
    """
    dataset_dir = 'Partial_REID'
    dataset_name = "PartialREID"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir, "upper_body_images")

        required_files = [
            self.dataset_dir,
        ]
        self.check_before_run(required_files)

        train = self._process_dir(self.dataset_dir, is_train=True)
        query = []
        gallery = []

        super(PartialREID, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_(\d*)')

        dataset = []
        for img_path in img_paths:
            pid, cam_id = map(int, pattern.search(img_path).groups())
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
class PDukeMTMC(ImageDataset):
    """
    train/test data

    id1
        image.jpg
    id2
        image.jpg
    """
    dataset_dir = 'pdukemtmc'
    dataset_name = "PDukeMTMC"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train/occluded_body_images')
        self.query_dir = osp.join(self.dataset_dir, 'test/occluded_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'test/whole_body_images')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self._process_dir(self.train_dir)
        query = self._process_dir(self.query_dir, is_train=False, is_query=True)
        gallery = self._process_dir(self.gallery_dir, is_train=False)

        super(PDukeMTMC, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, is_train=True, is_query=False):
        cls_paths = list(glob.glob(f"{dir_path}/*"))

        dataset = []
        for pid, cls_path in enumerate(cls_paths):
            for img_path in list(glob.glob(f"{cls_path}/*.jpg")):

                if is_train:
                    dataset.append((
                        img_path,
                        self.dataset_name + "_" + str(pid),
                        self.dataset_name + "_" + str(0)
                    ))
                else:
                    cam_id = 0 if is_query else -1
                    dataset.append((img_path, pid, cam_id))
                

        return dataset



@DATASET_REGISTRY.register()
class Pharmacity(ImageDataset):
    """
    only train data

    id1
        image.jpg
    id2
        image.jpg
    """
    dataset_dir = 'pharmacity_sup'
    dataset_name = "Pharmacity"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        train = self._process_dir(self.dataset_dir)
        query = []
        gallery = []

        super(Pharmacity, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, is_train=True, is_query=False):
        cls_paths = list(glob.glob(f"{dir_path}/*"))

        dataset = []
        for pid, cls_path in enumerate(cls_paths):
            for img_path in list(glob.glob(f"{cls_path}/*.jpg")):

                if is_train:
                    dataset.append((
                        img_path,
                        self.dataset_name + "_" + str(pid),
                        self.dataset_name + "_" + str(0)
                    ))
                else:
                    cam_id = 0 if is_query else -1
                    dataset.append((img_path, pid, cam_id))

        return dataset