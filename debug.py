import onnx
     
model = onnx.load('logs/PMC/pmc_09062023/onnx_model/pmc_onnx.onnx')
onnx.checker.check_model(model)