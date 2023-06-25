# import cv2
# import torch
# import sys
# sys.path.append('.')
# from fastreid.utils.visualizer import Visualizer
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor

# # Load config file
# cfg = get_cfg()
# cfg.merge_from_file("logs/PMC/pmc_09062023/config.yaml")  # Đường dẫn đến file cấu hình config.yaml
# cfg.MODEL.WEIGHTS = "logs/PMC/pmc_09062023/model_best.pth"  # Đường dẫn đến file trọng số model.pth
# cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Create predictor
# predictor = DefaultPredictor(cfg)

# # Load video
# video_path = "/mnt/nvme0n1/phuongnam/ByteTrack/videos/palace.mp4"  # Đường dẫn đến file video.mp4
# video = cv2.VideoCapture(video_path)

# fps = video.get(cv2.CAP_PROP_FPS)
# frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# codec = cv2.VideoWriter_fourcc(*"mp4v")
# # Create output video writer
# output_path = "output_video.mp4"  # Đường dẫn đến file video kết quả output_video.mp4
# output_video = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height))

# while True:
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Perform inference
#     outputs = predictor(frame)

#     # Visualize the predictions
#     visualizer = Visualizer(frame[:, :, ::-1], metadata=cfg.DATASETS.TEST)
#     vis_frame = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
#     vis_frame = vis_frame.get_image()

#     # Display the frame
#     output_video.write(vis_frame)
#     # cv2.imshow("Video", vis_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release video capture and close windows
# video.release()
# output_video.release()
# cv2.destroyAllWindows()

import numpy as np
import torch
# Đường dẫn đến hai tệp npy chứa đặc trưng của hai người
# feature_file_1 = "demo_output/1.npy"
# feature_file_2 = "demo_output/5.npy"
# feature_file_3 = "demo_output/10.npy"
# feature_file_4 = "demo_output/2.npy"# test
# feature_file_5 = "demo_output/29.npy"
# feature_file_6 = "demo_output/31.npy" # test

# # Đọc tệp npy và lấy đặc trưng của hai người
# feature_1 = torch.from_numpy(np.load(feature_file_1))
# feature_2 = torch.from_numpy(np.load(feature_file_2))
# feature_3 = torch.from_numpy(np.load(feature_file_3))
# feature_4 = torch.from_numpy(np.load(feature_file_4))
# feature_5 = torch.from_numpy(np.load(feature_file_5))
# feature_6 = torch.from_numpy(np.load(feature_file_6))
# print(1.0-feature_1)
# query = []
# gallery = []
# query.append(feature_1)
# query.append(feature_2)
# query.append(feature_3)
# query.append(feature_5)
# query = torch.cat(query, dim=0)
# gallery.append(feature_4)
# gallery.append(feature_6)
# gallery = torch.cat(gallery, dim=0)

# distmat = 1 - torch.mm(query, gallery.t())
# distmat = distmat.numpy()
# indices = np.argsort(distmat, axis=1)
# res = indices[:, 0]

# n_gallery = indices.shape[1]
# n_query = indices.shape[0]

# res = matrix = np.full((n_gallery, n_query), n_gallery+1)

# for i in range(n_query):
#     for j in range(n_gallery):
#         res[indices[i][j]][i] = min(j, res[indices[i][j]][i])
# # print(res)
# min_indices = np.nanargmin(res, axis=1)
# print(min_indices)

# # Tính toán khoảng cách Euclidean giữa hai đặc trưng
# distance = np.linalg.norm(feature_5 - feature_6)
# print(distance)
# # Xác định ngưỡng khoảng cách để quyết định hai người có giống nhau hay không
# threshold = 0.6  # Ngưỡng khoảng cách

# # Kiểm tra xem hai người có giống nhau hay không
# if distance < threshold:
#     print("Hai người giống nhau")
# else:
#     print("Hai người khác nhau")

# import numpy as np

# indices = np.array([[2, 0, 1, 3, 4, 6, 5],
#        [3, 4, 6, 2, 1, 5, 0],
#        [6, 5, 4, 3, 1, 2, 0]])

# n_gallery = indices.shape[1]
# n_query = indices.shape[0]

# res = matrix = np.full((n_gallery, n_query), n_gallery+1)

# for i in range(n_query):
#     for j in range(n_gallery):
#         res[indices[i][j]][i] = min(j, res[indices[i][j]][i])
# # print(res)
# min_indices = np.nanargmin(res, axis=1)
# print(min_indices)
    

feature_pth = np.load("onnx_output/gallery_1.npy") 
feature_onnx = np.load("demo_output/gallery_1.npy")
# feature_trt = np.load("null")
print("onnx và pth: ",np.testing.assert_allclose(feature_onnx, feature_pth, rtol=1e-3, atol=1e-6))
# print("trt và pth: ",np.testing.assert_allclose(feature_trt, feature_pth, rtol=1e-3, atol=1e-6))