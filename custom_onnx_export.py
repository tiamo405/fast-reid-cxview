import logging
import os
import argparse
import io
import sys

import onnx
# import onnxoptimizer
import torch
# from onnxsim import simplify
from torch.onnx import OperatorExportTypes

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.file_io import PathManager
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.logger import setup_logger

# import some modules added in project like this below
# sys.path.append("projects/FastDistill")
# from fastdistill import *

setup_logger(name="fastreid")
logger = logging.getLogger("fastreid.onnx_export")

setup_logger(name="fastreid")
logger = logging.getLogger("fastreid.onnx_export")


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Convert Pytorch to ONNX model")

    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--name",
        default="baseline",
        help="name for converted model"
    )
    parser.add_argument(
        "--output",
        default='onnx_model',
        help='path to save converted onnx model'
    )
    parser.add_argument(
        '--batch-size',
        default=1,
        type=int,
        help="the maximum batch size of onnx runtime"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--opset", type=int, default=11,
        help="ONNX opset version to generate models with.")
    return parser

def main(model, input):

    input_names = [ "actual_input_1" ] #+ [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]

    # Fixed Shape
    # torch.onnx.export(model, input, "alexnet_fixed.onnx", verbose=True, opset_version=args.opset,
    #                 input_names=input_names, output_names=output_names)

    # Dynamic Shape
    dynamic_axes = {"actual_input_1":{0:"batch_size"}, "output1":{0:"batch_size"}}
    print(dynamic_axes)
    torch.onnx.export(model, input, "onnx_dynamic.onnx", verbose=True, opset_version=args.opset,
                    input_names=input_names, output_names=output_names,
                    dynamic_axes=dynamic_axes)
    

if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    if cfg.MODEL.HEADS.POOL_LAYER == 'FastGlobalAvgPool':
        cfg.MODEL.HEADS.POOL_LAYER = 'GlobalAvgPool'
    model = build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    if hasattr(model.backbone, 'deploy'):
        model.backbone.deploy(True)
    model.eval()
    logger.info(model)

    inputs = torch.randn(args.batch_size, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]).to(model.device)
    onnx_model = main(model, inputs)

    # model_simp, check = simplify(onnx_model)

    # model_simp = remove_initializer_from_input(model_simp)

    # assert check, "Simplified ONNX model could not be validated"

    # PathManager.mkdirs(args.output)

    # save_path = os.path.join(args.output, args.name+'.onnx')
    # onnx.save_model(onnx_model, save_path)
    # logger.info("ONNX model file has already saved to {}!".format(save_path))