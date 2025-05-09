## Hardware-Accelerated Pruning for Efficient Vision Transformers

### Environment setup
```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
cd ..
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e .
pip install timm==0.4.9 einops
```

### Datasets
Download the validation set (images and annotations) of MS COCO dataset from [here](https://cocodataset.org/#download).


### Pretrained models
The pretrained model can be downloaded [here](https://github.com/ViTAE-Transformer/ViTPose/tree/main).


### Accuracy evaluation 

```
cd ViTPose/tools
python tools/test.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py /scratch/gilbreth/oprabhun/efficient_vitpose/orig_pretrained_models/vitpose-b.pth
```
The `infer.py` script accepts the following command-line arguments:

- Path to the config file for a given model variant. For example, base model:
  **Example:** `configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py`

- Path to the model weights file.  






