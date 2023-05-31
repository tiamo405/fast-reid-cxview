
##train

CUDA_VISIBLE_DEVICES=0 FASTREID_DATASETS="/mnt/nvme0n1/datasets/reid/" python tools/train_net.py \
  --config-file configs/Pharmacity/sbs_osnet.yml MODEL.DEVICE "cuda:0"


# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
#   --config-file ./configs/Market1501/sbs_R50.yml \
#    MODEL.DEVICE "cuda:0"


# python demo/demo.py --config-file logs/market1501/bagtricks_R50_AF/config.yaml \
#                     --parallel \
#                     --input "/mnt/nvme0n1/phuongnam/secret-reid/20220721_images_split_rotate/0/0_000011_0001.jpg" "/mnt/nvme0n1/phuongnam/secret-reid/20220721_images_split_rotate/0/0_000031_0001.jpg" "/mnt/nvme0n1/phuongnam/secret-reid/20220721_images_split_rotate/0/0_000033_0001.jpg" \
#                     --opts MODEL.WEIGHTS logs/market1501/sbs_osnet/model_final.pth

# CUDA_VISIBLE_DEVICES=0 python demo/visualize_result.py \
#   --config-file logs/market1501/sbs_osnet_PMC/config.yaml \
#   --parallel --vis-label --dataset-name "PMC"  \
#   --output output/market1501/sbs_osnet_PMC \
#   --opts MODEL.WEIGHTS logs/market1501/sbs_osnet_PMC/model_best.pth


# python demo/plot_roc_with_pickle.py 