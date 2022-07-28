import os, shutil
import torch
from torch.utils.data import Dataset
import pandas as pd

def export(export_dir: str, model, data: Dataset):
    print(f"Exporting to {export_dir}")
    
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)

    for dir_idx in range(len(data)):
        curdir = os.path.join(export_dir, str(dir_idx + 1))
        os.makedirs(curdir)

        item = data[dir_idx]

        for tar_station_idx in range(item["tar_locs"].size(0)):
            predicted = model.predict(item["features"], item["src_locs"], item["tar_locs"][tar_station_idx])

            df = pd.DataFrame.from_dict({"PM2.5": predicted.tolist()})

            df.to_csv(os.path.join(curdir, f"res_{dir_idx + 1}_{tar_station_idx + 1}.csv"), index=False)