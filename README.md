img shape error needs in `results['img_shape'] = img.shape[:2]` replace line 102 `img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)` with `img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend="pillow")` of file transform/loading.py in mmcv
