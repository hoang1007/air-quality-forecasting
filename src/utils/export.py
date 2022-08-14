import os
import shutil
from torch.utils.data import Dataset
import pandas as pd
import torch

from .functional import scale_softmax


def export(export_dir: str, model, data: Dataset, correlations: torch.Tensor):
    print(f"Exporting to {export_dir}")
    
    correlations = correlations.to(model.device)

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)

    for i in range(len(data)):
        item = data[i]

        folder_idx = item["folder_idx"]
        curdir = os.path.join(export_dir, str(folder_idx))
        os.makedirs(curdir)

        src_preds = model.predict(item)

        for tar_station_idx in range(item["tar_locs"].size(0)):
            weight_ = scale_softmax(correlations[tar_station_idx]).unsqueeze(-1)

            pred = (src_preds * weight_).sum(0)

            df = pd.DataFrame.from_dict({"PM2.5": pred.tolist()})

            df.to_csv(os.path.join(
                curdir, f"res_{folder_idx}_{tar_station_idx + 1}.csv"), index=False)