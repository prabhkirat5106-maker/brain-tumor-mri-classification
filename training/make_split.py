"""Write the notebook's exact train/val/test split to training/split.json.

    python training/make_split.py "C:/path/to/brain-tumor-mri-dataset"

Same steps as brain_mri_cnn.ipynb: list folders, drop exact duplicates (MD5) keeping the
first copy, then a stratified 70/15/15 split with random_state=42.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

data_path = sys.argv[1]
rows = []
for folder in os.listdir(data_path):
    for name in os.listdir(os.path.join(data_path, folder)):
        rows.append({"path": os.path.join(folder, name), "label": folder})
df = pd.DataFrame(rows)
df["hash"] = [hashlib.md5(open(os.path.join(data_path, p), "rb").read()).hexdigest() for p in df["path"]]
clean = df.drop_duplicates(subset="hash")
train, temp = train_test_split(clean, test_size=0.3, random_state=42, stratify=clean["label"])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])
out = {name: [[p.replace("\\", "/"), l] for p, l in zip(s["path"], s["label"])]
       for name, s in [("train", train), ("val", val), ("test", test)]}
Path(__file__).with_name("split.json").write_text(json.dumps(out))
print({k: len(v) for k, v in out.items()})
