# Introduction

**base:** https://github.com/JDAI-CV/fast-reid

# Changelogs

**2022/10/14:** Init repo
- Custom dataset (canifa, genviet, etc) is supported
- update osnet config
- update data folder struct


# Prepare Datasets

$FASTREID_DATASETS
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
├── tokyolife_sup
│   └── cls1
│   └── cls2
│   └── ...
├── canifa_sup
│   └── cls1
│   └── cls2
│   └── ...
└── genviet_sup
    └── cls1
    └── cls2
    └── ...

# Train

```bash
FASTREID_DATASETS="/content/datasets" python tools/train_net.py \
  --config-file configs/Market1501/sbs_osnet.yml MODEL.DEVICE "cuda:0"
  # MODEL.WEIGHTS "/content/drive/MyDrive/colab/reid/fast-reid/logs/market1501/model_final_osnet.pth"
```

# Export ONNX

```bash
python tools/deploy/onnx_export.py \
  --config-file configs/Market1501/sbs_osnet.yml \
  --output outputs/onnx_model --name model_best_osnet --opts MODEL.WEIGHTS weights/model_best_osnet.pth
```

# DataZoo

GoogleDrive: https://drive.google.com/drive/folders/1HQRqwRVhfVBazYty3IVescKmnkWyLq7T?usp=sharing

| Name          | train_ids | train_images | cameras |
| ------------- | :-------: | :----------: | :-----: |
| market1501    |    751    |    12936     |    6    |
| canifa_sup    |    230    |     4757     |    5    |
| tokyolife_sup |    345    |     6855     |    1    |
| genviet_sup   |    117    |     1312     |    2    |

**Note**: **_sup data* is the result of clustering from some UDA repos (SECRET, IDM, PPLR, etc)