from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile
import logging
import os
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

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
        self.dataset_dir = osp.join(self.root, self.dataset_dir, self.dataset_name)

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
class Pharmacity(ImageDataset):

    dataset_dir = 'PMC_sup_nam'
    dataset_name = "PMC"

    def __init__(self, root='datasets/', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self._process_dir(self.train_dir)
        query =self._process_dir_test(self.query_dir, is_query= True)
        gallery =self._process_dir_test(self.gallery_dir, is_query= False)

        super(Pharmacity, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, is_train=True):
        dataset = []
        cam_id = {}
        for folder_sup in list(glob.glob(f"{dir_path}/*")):
            name_folder = str(folder_sup).split('/')[-1]
            cls_paths = list(glob.glob(f"{folder_sup}/*"))
            for pid, cls_path in enumerate(cls_paths):
                for img_path in list(glob.glob(f"{cls_path}/*.jpg")):
                    try:
                        cam_id = int(osp.basename(img_path).split("_")[0])
                    except:
                        cam_id = 0
                    if is_train:
                        dataset.append((
                            img_path,
                            self.dataset_name +"_"+ name_folder + "_" + str(pid),
                            self.dataset_name + "_" + str(cam_id)
                        ))
                    else:
                        dataset.append((img_path, pid, cam_id))
                
        return dataset
    
    def _process_dir_test(self, dir_path, is_query = True): 
        dataset = []
        cls_paths = list(glob.glob(f"{dir_path}/*"))
        for pid, cls_path in enumerate(cls_paths):
            for img_path in list(glob.glob(f"{cls_path}/*.jpg")):

                if is_query:
                    cam_id = 0
                else :
                    cam_id = 1
                dataset.append((img_path, pid, cam_id))
                
        return dataset
    
@DATASET_REGISTRY.register()
class OccludedREID(ImageDataset):
    """
    only train data

    id1
        image.jpg
    id2
        image.jpg
    """
    dataset_dir = 'Occluded_REID'
    dataset_name = "OccludedREID"

    def __init__(self, root='datasets/', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, 'occluded_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'whole_body_images')

        required_files = [
            self.dataset_dir,
            # self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        # train = self._process_dir(self.train_dir)
        train = []
        query =self._process_dir_test(self.query_dir)
        gallery =self._process_dir_test(self.gallery_dir)

        super(OccludedREID, self).__init__(train, query, gallery, **kwargs)

    def _process_dir_test(self, dir_path): 
        dataset = []
        cls_paths = list(glob.glob(f"{dir_path}/*"))
        for pid, cls_path in enumerate(cls_paths):
            for img_path in list(glob.glob(f"{cls_path}/*")):
            # for img_path in list(glob.glob(f"{cls_path}/*.jpg")):
                if 'occluded_body_images' in dir_path:
                    cam_id = 0
                else :
                    cam_id = 1
                dataset.append((img_path, pid, cam_id))
                
        return dataset
    

@DATASET_REGISTRY.register()
class P_ETHZ(ImageDataset):
    """
    only train data

    id1
        image.jpg
    id2
        image.jpg
    """
    dataset_dir = 'P_ETHZ'
    dataset_name = "pethz"

    def __init__(self, root='datasets/', **kwargs):
        self.root = root
        self.train_dir = osp.join(self.root, self.dataset_dir)
        
        required_files = [
            self.train_dir,
        ]
        self.check_before_run(required_files)

        train = self._process_dir(self.train_dir)
        query = []
        gallery = []

        super(P_ETHZ, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path): 
        dataset = []
        whole_dir = os.path.join(dir_path, "whole_body_images")
        occluded_dir = os.path.join(dir_path, "occluded_body_images")
        cls = os.listdir(whole_dir)

        for pid, cls_id in enumerate(cls):
            cls_path_whole = os.path.join(whole_dir, cls_id)
            for img_path in list(glob.glob(f"{cls_path_whole}/*")):
                camid = 0
                p_id = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
                dataset.append((img_path, p_id, camid))
                
            cls_path_occluded = os.path.join(occluded_dir, cls_id)
            for img_path in list(glob.glob(f"{cls_path_occluded}/*")):
                camid = 0
                p_id = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
                dataset.append((img_path, p_id, camid))
            
        return dataset


@DATASET_REGISTRY.register()
class P_DukeMTMC_reid(ImageDataset):
    """
    only train data

    id1
        image.jpg
    id2
        image.jpg
    """
    dataset_dir = 'P-DukeMTMC-reid'
    dataset_name = "P_DukeMTMC_reid"

    def __init__(self, root='datasets/', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, "test", 'occluded_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, "test", 'whole_body_images')
        
        required_files = [
            self.train_dir,
        ]
        self.check_before_run(required_files)

        train = self._process_dir(self.train_dir)
        query = self._process_dir_test(self. query_dir)
        gallery = self._process_dir_test(self.gallery_dir)

        super(P_DukeMTMC_reid, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path): 
        dataset = []
        whole_dir = os.path.join(dir_path, "whole_body_images")
        occluded_dir = os.path.join(dir_path, "occluded_body_images")
        cls = os.listdir(whole_dir)

        for pid, cls_id in enumerate(cls):
            cls_path_whole = os.path.join(whole_dir, cls_id)
            for img_path in list(glob.glob(f"{cls_path_whole}/*")):
                camid = 0
                p_id = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
                dataset.append((img_path, p_id, camid))
                
            cls_path_occluded = os.path.join(occluded_dir, cls_id)
            for img_path in list(glob.glob(f"{cls_path_occluded}/*")):
                camid = 0
                p_id = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
                dataset.append((img_path, p_id, camid))
            
        return dataset
    
    def _process_dir_test(self, dir_path):
        dataset = []
        cls_paths = list(glob.glob(f"{dir_path}/*"))
        for pid, cls_path in enumerate(cls_paths):
            for img_path in list(glob.glob(f"{cls_path}/*")):
                if 'occluded_body_images' in dir_path:
                    cam_id = 0
                else :
                    cam_id = 1
                dataset.append((img_path, pid, cam_id))
                
        return dataset
    
        