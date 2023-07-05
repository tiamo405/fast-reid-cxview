
##train
CUDA_VISIBLE_DEVICES=0 FASTREID_DATASETS="data" python tools/train_net.py \
  --config-file configs/Pharmacity/sbs_osnet.yml MODEL.DEVICE "cuda:0"


# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
#   --config-file ./configs/Market1501/sbs_R50.yml \
#    MODEL.DEVICE "cuda:0"


# python demo/demo.py --config-file logs/PMC/pmc_09062023/config.yaml \
#                     --parallel \
#                     --input "demo/gallery_1.jpg" \
#                     --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth

# CUDA_VISIBLE_DEVICES=0 FASTREID_DATASETS="data" python demo/visualize_result.py \
#   --config-file logs/PMC/pmc_09062023/config.yaml \
#   --parallel --vis-label --dataset-name "Pharmacity"  \
#   --output output/pmc/sbs_osnet_PMC_test \
#   --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth


# CUDA_VISIBLE_DEVICES=1 python demo/test.py --config-file logs/PMC/pmc_09062023/config.yaml \
#                     --parallel \
#                     --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth

# export onnx
# python tools/deploy/onnx_export.py --config-file logs/PMC/pmc_09062023/config.yaml \
#   --name pmc_onnx --output logs/PMC/pmc_09062023/onnx_model \
#   --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth \



# test model onnx
# python tools/deploy/onnx_inference.py --height 384 --width 128\
#   --model-path logs/PMC/pmc_09062023/onnx_model/pmc_onnx.onnx \
#  --input "demo/gallery_1.jpg" --output onnx_output




# export trt
# ./export pmc_onnx.onnx model.plan

# so sánh kết quả khi export
# python demo/sosanhweight.py


# file .onnx
# wget --no-check-certificate -O pmc_onnx.onnx "https://drive.google.com/uc?export=download&id=1hO8cDHkqKBIy8jN0Q6SUo6X1adw1ktih"

# docker
#docker run --gpus device=0 --name fastreid --runtime nvidia -dit -v /mnt/nvme0n1/phuongnam/fast-reid:/workspace nvcr.io/nvidia/pytorch:20.03-py3

# test model TRT
# ./trtexec --loadEngine=/workspace/cpp/build/model.plan --explicitBatch --shapes=input_0:1x3x384x128