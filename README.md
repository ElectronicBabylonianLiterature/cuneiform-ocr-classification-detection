## Two-Stage Cuneiform Sign Detection (Detecting Bounding Boxes + Image Classification)

This Repository based on mmocr and mmcv can be used for training the Model once trained one has to use a different
repository (https://github.com/ElectronicBabylonianLiterature/cuneiform-ocr) based on mmdetection to unify both models to get the final model.
This is due to dependencies of the two repositories. For the data use https://github.com/ElectronicBabylonianLiterature/cuneiform-ocr-data.

The checkpoints of the pretrained models can be found here: https://syncandshare.lrz.de/getlink/fi39rfQ11LtbxEBTmwVs2u/
There should be folder `cuneiform-ocr-classification-detection/checkpoints/` with all checkpoints


### Requirements (Tested with Python 3.11 )
- torch=="2.0.1", torchvision
- pip install -U openmim 
- mim install mmengine (tested with 0.8.3)
- mim install "mmcls==1.0.0rc5" (installing as a dependency "mmcv==2.0.0")
- mim install "mmdet==3.0.0rc6"
- mim install mmocr #tested with version 1.0.1
- pip install -r requirements.txt


### Detection
#### Training
- cd cuneiform_ocr/detection
- have data in cuneiform_ocr/detection/data
- data
  - icdar2015
    - textdet_imgs
    - textdet_test.json
    - textdet_train.json

- python3 mmocr_tools/train.py custom_configs/fcenet_dcvn_debug.py #validation set is used after one epoch to make sure everything is working
- python3 mmocr_tools/train.py custom_configs/fcenet_dcvn.py #validation set is used after 50 epochs
#### Testing
- cd cuneiform_ocr/detection
- python3 mmocr_tools/test.py custom_configs/fcenet_dcvn.py ../../checkpoints/fcenet_resnet50-dcnv2.pth  (replace checkpoints with trained checkpoints)

### Classification
#### Training
- cd cuneiform_ocr/classification
- have data in cuneiform_ocr/classification/data
- data
  - ebl
    - test_set
    - train_set
    - test.txt
    - train.txt
    - classes.txt (optional)
- python3 mmclassification_tools/train.py custom_configs/efficient_net.py

#### Testing
- python3 mmclassification_tools/test.py custom_configs/efficient_net.py ../../checkpoints/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth

# Errors
Sometimes trying to install mim install "mmcv==2.0.0rc4" can take forever and it fails
if the pytoch version and cuda mismatch. In this case skip installation of mmcv and install mim install "mmcls==1.0.0rc5"
which as a dependency has mmcv and will install it for you. (You can try using torch cpu installing everything
and then uninstall torch cpu and install torch with cuda to dotch the pytorch version and cuda mismatch error).
Now when running a script you may get this error:
```python
Traceback (most recent call last):
  File "/home/yunus/PycharmProjects/cuneiform-ocr-classification-detection/cuneiform_ocr/classification/mmclassification_tools/train.py", line 12, in <module>
    from mmcls.utils import register_all_modules
  File "/home/yunus/PycharmProjects/cuneiform-ocr-classification-detection/.venv/lib/python3.11/site-packages/mmcls/__init__.py", line 18, in <module>
    and mmcv_version < digit_version(mmcv_maximum_version)), \
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

In which case just comment the assert statement in `.venv/lib/python3.11/site-packages/mmcls/__init__.py` and 
everything should work.

img shape error needs in `results['img_shape'] = img.shape[:2]` replace line 102 `img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)` with `img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend="pillow")` of file .venv/lib/python3.11/site-packages/mmcv/transforms/loading.py in mmcv

