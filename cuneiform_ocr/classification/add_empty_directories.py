import shutil
from pathlib import Path

classes = ['ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ461', 'ABZ142', 'ABZ480', 'ABZ1', 'ABZ231', 'ABZ533', 'ABZ449', 'ABZ318', 'ABZ75', 'ABZ61', 'ABZ354', 'ABZ139', 'ABZ381', 'ABZ597', 'ABZ536', 'ABZ308', 'ABZ330', 'ABZ328', 'ABZ86', 'ABZ15', 'ABZ214', 'ABZ545', 'ABZ73', 'ABZ295', 'ABZ55', 'ABZ335', 'ABZ371', 'ABZ151', 'ABZ457', 'ABZ537', 'ABZ69', 'ABZ353', 'ABZ68', 'ABZ5', 'ABZ296', 'ABZ84', 'ABZ366', 'ABZ411', 'ABZ396', 'ABZ206', 'ABZ58', 'ABZ324', 'ABZ376', 'ABZ99', 'ABZ384', 'ABZ59', 'ABZ532', 'ABZ334', 'ABZ589', 'ABZ383', 'ABZ343', 'ABZ586', 'ABZ399', 'ABZ74', 'ABZ211', 'ABZ145', 'ABZ7', 'ABZ212', 'ABZ78', 'ABZ367', 'ABZ38', 'ABZ319', 'ABZ85', 'ABZ115', 'ABZ322', 'ABZ97', 'ABZ144', 'ABZ112', 'ABZ427', 'ABZ207', 'ABZ60', 'ABZ79', 'ABZ80', 'ABZ232', 'ABZ142a', 'ABZ312', 'ABZ52', 'ABZ331', 'ABZ128', 'ABZ314', 'ABZ535', 'ABZ575', 'ABZ134', 'ABZ465', 'ABZ167', 'ABZ172', 'ABZ339', 'ABZ6', 'ABZ331e+152i', 'ABZ306', 'ABZ12', 'ABZ2', 'ABZ148', 'ABZ397', 'ABZ554', 'ABZ570', 'ABZ441', 'ABZ147', 'ABZ472', 'ABZ230', 'ABZ440', 'ABZ104', 'ABZ595', 'ABZ455', 'ABZ313', 'ABZ298', 'ABZ62', 'ABZ412', 'ABZ468', 'ABZ101', 'ABZ111', 'ABZ483', 'ABZ538', 'ABZ471', 'ABZ87', 'ABZ143', 'ABZ565', 'ABZ152', 'ABZ205', 'ABZ72', 'ABZ406', 'ABZ138', 'ABZ50', 'ABZ401', 'ABZ307', 'ABZ126', 'ABZ124', 'ABZ164', 'ABZ529', 'ABZ559', 'ABZ94', 'ABZ56', 'ABZ437', 'ABZ393', 'ABZ398']

path = Path("data/ebl/test_set/test_set")

# iter through list classes and if not in path, create directory
for c in classes:
    if not (path / c).exists():
        (path / c).mkdir()
        print(f"Created directory {c}")
    else:
        print(f"Directory {c} already exists")

# delete all directories not in classes
for d in path.iterdir():
    if d.name not in classes:
        # delete dictory recursively
        shutil.rmtree(d)
        print(f"Deleted directory {d.name}")
    else:
        print(f"Directory {d.name} already exists")


