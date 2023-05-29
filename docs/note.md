1. Occlusion Challenges

1.1. Backbone: OSNet
- Train Dataset: Market1501, Canifa, Genviet, Tokyolife, <br>
DukeMTMC-reid, P-DukeMTMC-reid, Partial_REID, Pharmacity (381 ids)
- Test Dataset: OccludedREID

| Methods  |  Rank-1 | mAP  |
|---|---|---|
| FED  | **86.3**  | **79.3**  |
| TransReID  | 70.2  | 67.3  |
| BPBreID  | 76.9  | 68.6  |
| HOReID  |  80.3 | 70.2  |
| VGTri  | 81.0  |  71.0 |
| **Ours**  | 78.8  | 74.54  |

1.2. Backbone: Resnet50

TODO

1. Normal Challenges

2.1. Backbone: OSNet
- Train Dataset: Market1501, P-DukeMTMC-reid, Partial_REID, Pharmacity
- Test Dataset: Market1501

| Methods  |  Rank-1 | mAP  |
|---|---|---|
| **Ours**  | 92.9  | 81.9  |

2.2. Backbone: Resnet50

TODO