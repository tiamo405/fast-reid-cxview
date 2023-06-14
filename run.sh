
##train

# CUDA_VISIBLE_DEVICES=0 FASTREID_DATASETS="/mnt/nvme0n1/datasets/reid/" python tools/train_net.py \
#   --config-file configs/Pharmacity/sbs_osnet.yml MODEL.DEVICE "cuda:0"


# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
#   --config-file ./configs/Market1501/sbs_R50.yml \
#    MODEL.DEVICE "cuda:0"


# python demo/demo.py --config-file logs/PMC/pmc_09062023/config.yaml \
#                     --parallel \
#                     --input "/mnt/nvme0n1/datasets/reid/PMC_sup_nam/query/0/0_000001.jpg" \
#                      "/mnt/nvme0n1/datasets/reid/PMC_sup_nam/query/0/0_000005.jpg" \
#                      "/mnt/nvme0n1/datasets/reid/PMC_sup_nam/query/0/0_000010.jpg" \
#                      "/mnt/nvme0n1/datasets/reid/PMC_sup_nam/query/1/0_000002.jpg" \
#                      "/mnt/nvme0n1/datasets/reid/PMC_sup_nam/query/10/0_001929.jpg" \
#                      "/mnt/nvme0n1/datasets/reid/PMC_sup_nam/query/10/0_001931.jpg"\
#                     --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth

# CUDA_VISIBLE_DEVICES=1 python demo/visualize_result.py \
#   --config-file logs/PMC/pmc_09062023/config.yaml \
#   --parallel --vis-label --dataset-name "Pharmacity"  \
#   --output output/pmc/sbs_osnet_PMC \
#   --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth


CUDA_VISIBLE_DEVICES=1 python demo/test.py --config-file logs/PMC/pmc_09062023/config.yaml \
                    --parallel \
                    --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth

# export onnx
python tools/deploy/onnx_export.py --config-file logs/PMC/pmc_09062023/config.yaml \
  --name sbs_osnet_pmc --output logs/PMC/pmc_09062023/onnx_model \
  --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth

# test model onnx
python tools/deploy/onnx_inference.py --height 384 --width 128\
  --model-path logs/PMC/pmc_09062023/pmc_onnx.onnx \
 --input "demo/gallery_1.jpg" --output onnx_output