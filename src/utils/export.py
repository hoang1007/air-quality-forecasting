import os
import shutil
from torch.utils.data import Dataset
from model import BaseAQFModel
import pandas as pd
import torch

from .functional import scale_softmax


def export(export_dir: str, model: BaseAQFModel, data: Dataset):
    print(f"Exporting to {export_dir}")

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)

    for i in range(len(data)):
        item = data[i]

        folder_idx = item["folder_idx"]
        curdir = os.path.join(export_dir, str(folder_idx))
        os.makedirs(curdir)

        for tar_station_idx in range(item["tar_locs"].size(0)):
            predicted = model.predict(
                item["features"],
                item["src_locs"],
                item["tar_locs"][tar_station_idx],
                item["src_masks"])

            df = pd.DataFrame.from_dict({"PM2.5": predicted.tolist()})

            df.to_csv(os.path.join(
                curdir, f"res_{folder_idx}_{tar_station_idx + 1}.csv"), index=False)


def export_dqaff(export_dir: str, model: BaseAQFModel, data: Dataset):
    print(f"Exporting to {export_dir}")

    weights = torch.tensor([[-0.1833, -0.5908,  0.5385, -0.6062,  0.5575, -0.1952, -0.1376,  0.5791,
                             -0.3552, -0.4850,  0.4052],
                            [0.4392,  0.0281,  0.3730, -0.4298,  0.2668, -0.4788, -0.6369, -0.2180,
                             -0.0795, -0.4105,  0.3912],
                            [0.4081, -0.2197, -0.3096,  0.4927,  0.8725,  0.0762, -0.6778,  0.4540,
                             0.1712, -0.2912,  0.2438],
                            [0.7110, -0.3105,  0.0597, -0.7717, -0.5332,  0.1624, -0.3558, -0.3818,
                             0.3458,  0.1034,  0.1645]])

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
            weight_ = scale_softmax(weights[tar_station_idx]).unsqueeze(-1)

            pred = (src_preds * weight_).sum(0)

            df = pd.DataFrame.from_dict({"PM2.5": pred.tolist()})

            df.to_csv(os.path.join(
                curdir, f"res_{folder_idx}_{tar_station_idx + 1}.csv"), index=False)
