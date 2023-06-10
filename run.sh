
##train

# CUDA_VISIBLE_DEVICES=0 FASTREID_DATASETS="/mnt/nvme0n1/datasets/reid/" python tools/train_net.py \
#   --config-file configs/Pharmacity/sbs_osnet.yml MODEL.DEVICE "cuda:0"


# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
#   --config-file ./configs/Market1501/sbs_R50.yml \
#    MODEL.DEVICE "cuda:0"


python demo/demo.py --config-file logs/PMC/pmc_09062023/config.yaml \
                    --parallel \
                    --input "/mnt/nvme0n1/datasets/reid/PMC_sup_nam/query/0/0_000001.jpg" \
                     "/mnt/nvme0n1/datasets/reid/PMC_sup_nam/query/0/0_000005.jpg" \
                     "/mnt/nvme0n1/datasets/reid/PMC_sup_nam/query/0/0_000010.jpg" \
                     "/mnt/nvme0n1/datasets/reid/PMC_sup_nam/query/1/0_000002.jpg" \
                     "/mnt/nvme0n1/datasets/reid/PMC_sup_nam/query/10/0_001929.jpg" \
                     "/mnt/nvme0n1/datasets/reid/PMC_sup_nam/query/10/0_001931.jpg"\
                    --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth

# CUDA_VISIBLE_DEVICES=0 python demo/visualize_result.py \
#   --config-file logs/market1501/sbs_osnet_PMC/config.yaml \
#   --parallel --vis-label --dataset-name "PMC"  \
#   --output output/market1501/sbs_osnet_PMC \
#   --opts MODEL.WEIGHTS logs/market1501/sbs_osnet_PMC/model_best.pth


# python demo/demo_video.py --config-file logs/PMC/pmc_09062023/config.yaml \
#                     --parallel \
#                     --opts MODEL.WEIGHTS logs/PMC/pmc_09062023/model_best.pth