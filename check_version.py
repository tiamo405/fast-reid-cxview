import torch
import pycuda
import tensorrt
import onnx

print('version torch: ', torch.__version__)
print('version pycuda: ', pycuda.VERSION)
print('version tensorrt: ', tensorrt.__version__)
print('version onnx: ', onnx.__version__)