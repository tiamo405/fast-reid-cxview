
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
python tools/deploy/onnx_export.py --config-file logs/PMC/pmc_09062023/config.yaml \
  --name pmc_onnx --output logs/PMC/pmc_09062023/onnx_model \

#   --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth \
# python custom_onnx_export.py --config-file logs/PMC/pmc_09062023/config.yaml \
#   --name pmc_onnx --output logs/PMC/pmc_09062023/onnx_model \
#   --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth \



# test model onnx
# python tools/deploy/onnx_inference.py --height 384 --width 128\
#   --model-path logs/PMC/pmc_09062023/onnx_model/pmc_onnx.onnx \
#  --input "demo/gallery_1.jpg" --output onnx_output

# python tools/deploy/onnx_inference.py --height 384 --width 128\
#   --model-path pmc_onnx_2.onnx \
#  --input "demo/gallery_1.jpg" --output onnx_output

# docker run --gpus device=0 --name test --runtime nvidia -dit -v /mnt/nvme0n1/phuongnam/fast-reid:/workspace nvcr.io/nvidia/pytorch:20.03-py3

# python tools/deploy/trt_inference.py --model-path outputs/trt_model/pmc_trt.engine \
#   --input demo/gallery_1.jpg --batch-size 1 --height 384 --width 128 --output trt_output 

# export trt
# ./export pmc_onnx.onnx model.plan

# so sánh kết quả khi export
# python demo/sosanhweight.py

# trtexec --explicitBatch --onnx=logs/PMC/pmc_09062023/onnx_model/pmc_onnx.onnx \
#         --minShapes=data:1x3x384x128 \
#         --optShapes=data:3x3x384x128 \
#         --maxShapes=data:5x3x384x128 \
#         --shapes=data:3x3x384x128 \
#         --saveEngine=onnx_dynamic.engine
# nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb
# os="ubuntu1x04"
# tag="cudax.x-trt7.x.x.x-ga-yyyymmdd"
# sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb
# sudo apt-key add /var/nv-tensorrt-repo-cuda10.2-trt7.0.0.11-ga-20191216/7fa2af80.pub

# sudo apt-get update
# sudo apt-get install tensorrt

docker run --gpus device=0 --name fastreid-export --runtime nvidia -dit -v /mnt/nvme0n1/phuongnam/fast-reid:/workspace nvcr.io/nvidia/pytorch:20.03-py3
./trtexec --loadEngine=/workspace/cpp/build/model.plan --explicitBatch --shapes=input_0:1x3x384x128