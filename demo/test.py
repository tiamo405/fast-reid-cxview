# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
import os
import sys
import torch

import torch.nn.functional as F
import cv2
import numpy as np
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager

from predictor import FeatureExtractionDemo

# import some modules added in project like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser
class reID():
    def __init__(self, args, parallel=False) :
        self.args = args
        self.cfg = setup_cfg(self.args)
        # self.parallel = cfg.parallel
        self.demo = FeatureExtractionDemo(self.cfg, parallel=args.parallel)
        
    def postprocess(self, features):
        # Normalize feature to compute cosine distance
        features = F.normalize(features)
        features = features.cpu().data.numpy()
        return features

    def feature(self, img) :
        feat = self.demo.run_on_image(img)
        feat = self.postprocess(feat)
        return feat

    def gallery_feat(self, dir_path) :
        ids =[]
        g_feat = []
        for file in list(glob.glob(f"{dir_path}/*")):
            id = os.path.basename(file).split('.')[0]
            feature = torch.from_numpy(np.load(file))
            g_feat.append(feature)
            ids.append(id)
        g_feat = torch.cat(g_feat, dim=0)
        return g_feat,  ids

    def save2galery(self, dir_path, feat, id):
        os.makedirs(dir_path, exist_ok= True)
        np.save(os.path.join(dir_path, str(id) + '.npy'), feat)
        
    def create_gallery(self, dir_path, dir_save):
        for file in list(glob.glob(f"{dir_path}/*")):
            id = os.path.basename(file).split('.')[0].split('_')[-1]
            img = cv2.imread(file)
            feat = self.feature(img, self.demo)
            self.save2galery(dir_save, feat, id)

    def find_id(self, feat) :
        query = []
        query.append(torch.from_numpy(feat))
        query = torch.cat(query, dim = 0)
        gallery, ids = self.gallery_feat("demo_output")
        distmat = 1 - torch.mm(query, gallery.t())
        distmat = distmat.numpy()
        indices = np.argsort(distmat, axis=1)
        q_id = ids[indices[:, 0][0]]
        
        return q_id

if __name__ == '__main__':
    
    args = get_parser().parse_args()
    # cfg = setup_cfg(args)
    reid = reID(args=args)

    PathManager.mkdirs(args.output)
    
    dir_path = "/mnt/nvme0n1/datasets/reid/PMC_sup_nam/gallery_test/"
    dir_save = "deno_output"
    # create_gallery(dir_path = dir_path, dir_save= dir_save)
    path_img = "demo/gallery_1.jpg"
    img = cv2.imread(path_img)
    feat = reid.feature(img)
    features = []
    features.append(feat)
    features.append(feat)
    features.append(feat)
    print(reid.find_id(feat))
    # print(features.shape)
    
    
    
    