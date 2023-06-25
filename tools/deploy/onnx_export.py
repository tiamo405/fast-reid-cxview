# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

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
    return parser


def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model


def export_onnx_model(model, inputs):
    """
    Trace and export a model to onnx format.
    Args:
        model (nn.Module):
        inputs (torch.Tensor): the model will be called by `model(*inputs)`
    Returns:
        an onnx model
    """
    assert isinstance(model, torch.nn.Module)

    # make sure all modules are in eval mode, onnx may change the training state
    # of the module if the states are not consistent
    def _check_eval(module):
        assert not module.training

    model.apply(_check_eval)

    logger.info("Beginning ONNX file converting")
    # Export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            input_names = ['input_1']
            output_names = ['output_1']
            dynamic_axes = {input_names[0]: {0:'batch'}}
            for _, name in enumerate(output_names):
                dynamic_axes[name] = dynamic_axes[input_names[0]]
            extra_args = {'opset_version': 10, 'verbose': False,
                'input_names': input_names, 'output_names': output_names,
                'dynamic_axes': dynamic_axes}
            torch.onnx.export(model, inputs, f, **extra_args)

            onnx_model = onnx.load_from_string(f.getvalue())

    # input_names = ['input_1']
    # output_names = ['output_1']
    # import io
    # onnx_bytes = io.BytesIO()
    # zero_input = inputs
    # # zero_input = zero_input.to(device)
    # dynamic_axes = {input_names[0]: {0:'batch'}}
    # for _, name in enumerate(output_names):
    #     dynamic_axes[name] = dynamic_axes[input_names[0]]
    # extra_args = {'opset_version': 10, 'verbose': False,
    #                 'input_names': input_names, 'output_names': output_names,
    #                 'dynamic_axes': dynamic_axes}
    # torch.onnx.export(model, zero_input, onnx_bytes, **extra_args)
    # with open('test.onnx', 'wb') as out:
    #     out.write(onnx_bytes.getvalue())

    return onnx_model


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
    onnx_model = export_onnx_model(model, inputs)
    # export_onnx_model(model, inputs)


    PathManager.mkdirs(args.output)

    save_path = os.path.join(args.output, args.name+'.onnx')
    onnx.save_model(onnx_model, save_path)
    logger.info("ONNX model file has already saved to {}!".format(save_path))
