
##train

# FASTREID_DATASETS="data" python tools/train_net.py \
#   --config-file configs/Pharmacity/sbs_osnet.yml MODEL.DEVICE "cuda:0"


# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
#   --config-file ./configs/Market1501/sbs_R50.yml \
#    MODEL.DEVICE "cuda:0"


# python demo/demo.py --config-file logs/PMC/pmc_09062023/config.yaml \
#                     --parallel \
#                     --input "demo/gallery_1.jpg" \
#                     --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth

# CUDA_VISIBLE_DEVICES=0 python demo/visualize_result.py \
#   --config-file logs/PMC/pmc_09062023/config.yaml \
#   --parallel --vis-label --dataset-name "Pharmacity"  \
#   --output output/pmc/sbs_osnet_PMC \
#   --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth


# CUDA_VISIBLE_DEVICES=1 python demo/test.py --config-file logs/PMC/pmc_09062023/config.yaml \
#                     --parallel \
#                     --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth

# export onnx
# python tools/deploy/onnx_export.py --config-file logs/PMC/pmc_09062023/config.yaml \
#   --name pmc_onnx --output logs/PMC/pmc_09062023/onnx_model \
#   --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth \
# python custom_onnx_export.py --config-file logs/PMC/pmc_09062023/config.yaml \
#   --name pmc_onnx --output logs/PMC/pmc_09062023/onnx_model \
#   --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth \



# test model onnx
# python tools/deploy/onnx_inference.py --height 384 --width 128\
#   --model-path test.onnx \
#  --input "demo/gallery_1.jpg" --output onnx_output

# docker run --runtime nvidia -itd --ipc=host --net=host --privileged --name reid-tensorrt -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/nvme0n1/phuongnam/fast-reid:/workspace nvcr.io/nvidia/pytorch:20.03-py3

# docker run --name fastreid-export --runtime nvidia -dit -v /mnt/nvme0n1/phuongnam/fast-reid:/workspace nvcr.io/nvidia/pytorch:20.03-py3

# python tools/deploy/trt_inference.py --model-path outputs/trt_model/pmc_trt.engine \
#   --input demo/gallery_1.jpg --batch-size 1 --height 384 --width 128 --output trt_output 

# export trt
# ./export pmc_onnx.onnx model.plan

# so sánh kết quả khi export
# python demo/sosanhweight.py

# trtexec --explicitBatch --onnx=nano_rotate_20220620.onnx \
#         --minShapes=data:1x3x480x480 \
#         --optShapes=data:3x3x480x480 \
#         --maxShapes=data:5x3x480x480 \
#         --shapes=data:3x3x480x480 \
#         --saveEngine=onnx_dynamic.engine