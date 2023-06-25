import onnx

# Đường dẫn đến file ONNX
onnx_file_path = "logs/PMC/pmc_09062023/onnx_model/pmc_onnx.onnx"

# Kiểm tra tính hợp lệ của file ONNX
model = onnx.load(onnx_file_path)
is_valid = onnx.checker.check_model(model)

if is_valid:
    print("ONNX file is valid.")
else:
    print("ONNX file is invalid.")


import onnxruntime as ort


# Kiểm tra tính hợp lệ của file ONNX bằng ONNX Runtime
try:
    model = onnx.load(onnx_file_path)
    onnx.checker.check_model(model)
    print("ONNX file is valid.")
except onnx.exceptions.InvalidModel as e:
    print("ONNX file is invalid:", e)

# Hoặc sử dụng Netron để xem cấu trúc của file ONNX
import netron

# Đường dẫn đến file ONNX
onnx_file_path = "path/to/your/model.onnx"

# Mở giao diện Netron với file ONNX
netron.start(onnx_file_path)
