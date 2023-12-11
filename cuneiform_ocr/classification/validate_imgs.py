from pathlib import Path
import os
import mmcv

mmcv.use_backend("pillow")
from PIL import Image

def _validate_img(file):
        try:
                    img = Image.open(file)
                            if mmcv.imread(file) is None:
                                            print(f"Could not read {file}")
                                                    k = mmcv.imfrombytes(open(file, "rb").read())
                                                            if k is None:
                                                                            print(f"Could not read {file}")
                                                                                    img.verify()
                                                                                            img.close()
                                                                                                except (IOError, SyntaxError):
                                                                                                            print("Bad file:", file)
                                                                                                                    os.remove(file)


                                                                                                                    def validate_imgs(imgs_path: Path):
                                                                                                                            for dir in imgs_path.iterdir():
                                                                                                                                        if dir.is_dir():
                                                                                                                                                        for sign_dir in dir.iterdir():
                                                                                                                                                                            for dir in sign_dir.iterdir():
                                                                                                                                                                                                    for file in dir.iterdir():
                                                                                                                                                                                                                                _validate_img(file)

                                                                                                                                                                                                                                if __name__ == "__main__":
                                                                                                                                                                                                                                        # can be used to manually validate files and delete bad ones for classification
                                                                                                                                                                                                                                            imgs_path = Path("cuneiform_ocr_data/classification/data/ebl")
                                                                                                                                                                                                                                                validate_imgs(imgs_path)

