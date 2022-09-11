import os
import shutil
from torch.utils.data import Dataset
import pandas as pd
import torch


# def export(export_dir: str, model, data: Dataset, correlations: torch.Tensor):
#     print(f"Exporting to {export_dir}")
    
#     correlations = correlations.to(model.device)

#     if os.path.exists(export_dir):
#         shutil.rmtree(export_dir)
#     os.makedirs(export_dir)

#     for i in range(len(data)):
#         item = data[i]

#         folder_name = item["folder_name"]
#         curdir = os.path.join(export_dir, folder_name)
#         os.makedirs(curdir)

#         src_preds = model.predict(item)

#         for tar_station_idx in range(item["tar_locs"].size(0)):
#             weight_ = scale_softmax(correlations[tar_station_idx]).unsqueeze(-1)

#             pred = (src_preds * weight_).sum(0)

#             df = pd.DataFrame.from_dict({"PM2.5": pred.tolist()})

#             df.to_csv(os.path.join(
#                 curdir, f"res_{folder_name}_{tar_station_idx + 1}.csv"), index=False)

#     print(f"Exported to {export_dir} successfully.")


def batch_export(export_dir: str, model, data: Dataset):
    print(f"Exporting to {export_dir}")

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)

    for i in range(len(data)):
        item = data[i]

        folder_name = item["folder_name"]
        curdir = os.path.join(export_dir, folder_name)
        os.makedirs(curdir)

        preds = model.predict(item)

        for tar_station_idx in range(item["tar_locs"].size(0)):
            df = pd.DataFrame.from_dict({"PM2.5": preds[tar_station_idx].tolist()})

            df.to_csv(os.path.join(
                curdir, f"res_{folder_name}_{tar_station_idx + 1}.csv"), index=False)

    print(f"Exported to {export_dir} successfully.")