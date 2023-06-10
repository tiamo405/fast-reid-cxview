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

# Đường dẫn đến hai tệp npy chứa đặc trưng của hai người
feature_file_1 = "demo_output/0_000001.npy"
feature_file_2 = "demo_output/0_000005.npy"
feature_file_3 = "demo_output/0_000010.npy"
feature_file_4 = "demo_output/0_000002.npy"
feature_file_5 = "demo_output/0_001929.npy"
feature_file_6 = "demo_output/0_001931.npy"

# Đọc tệp npy và lấy đặc trưng của hai người
feature_1 = np.load(feature_file_1)
feature_2 = np.load(feature_file_2)
feature_3 = np.load(feature_file_3)
feature_4 = np.load(feature_file_4)
feature_5 = np.load(feature_file_5)
feature_6 = np.load(feature_file_6)

# Tính toán khoảng cách Euclidean giữa hai đặc trưng
distance = np.linalg.norm(feature_5 - feature_6)
print(distance)
# Xác định ngưỡng khoảng cách để quyết định hai người có giống nhau hay không
threshold = 0.6  # Ngưỡng khoảng cách

# Kiểm tra xem hai người có giống nhau hay không
if distance < threshold:
    print("Hai người giống nhau")
else:
    print("Hai người khác nhau")

