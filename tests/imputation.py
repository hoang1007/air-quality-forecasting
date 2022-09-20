import shutil
from os import path
import numpy as np
import pandas as pd
import torch
from dataset import private_train_data
from dataset.preprocessing import idw_imputation, median_imputation, spline_imputation
from utils.metrics import MultiMetrics


class ImputationDataset:
    def __init__(
        self,
        max_seq_len: int = 20,
        extract_dir: str = "./data",
        archive_path: str = "../private-data.zip"
    ):
        self.max_seq_len = max_seq_len
        self.extract_dir = extract_dir

        if not path.exists(self.extract_dir):
            shutil.unpack_archive(archive_path, extract_dir=self.extract_dir)

    def get_dataset(self, mask_ratio: float = 0.2):
        np.random.seed(3107)

        raw_data = private_train_data(path.join(self.extract_dir, "train"))["air"]

        labels = {}

        for st_name, station in raw_data.items():
            seq = station["data"]["PM2.5"]
            assert isinstance(seq, pd.Series)

            nna_ids = np.argwhere(seq.notna().values).squeeze(-1)
            # keep_num = min(len(nna_ids), int(len(seq) * mask_ratio))
            keep_num = int(len(nna_ids) * mask_ratio)
            nna_ids = self._random_choice(nna_ids, keep_num, self.max_seq_len)

            labels[st_name] = (
                nna_ids,
                seq.values[nna_ids]
            )

            for idx in nna_ids:
                seq.at[idx] = np.nan
        
        return raw_data, labels

    def get_na_summary(self, raw_data):
        summary = {}
        for st_name, station in raw_data.items():
            summary[st_name] = station["data"]["PM2.5"].isna().sum()

        return pd.DataFrame(summary, index=[0])

    @staticmethod
    def _random_choice(seq: np.ndarray, num: int, max_seq_len: int):
        # keep_ids = np.random.permutation(len(seq))[:num]

        # seq = seq[keep_ids]
        # return seq
        group_ids = ImputationDataset._random_group(num, max_seq_len=max_seq_len)
        increment_vec = np.array([0] + group_ids)

        for i in range(1, len(increment_vec)):
            increment_vec[i] += increment_vec[i - 1]

        chosen_ids = None
        high = len(seq) - group_ids[-1]
        for i in range(len(group_ids) - 1, -1, -1):
            low = increment_vec[i]

            if low < high:
                shift_range = np.random.randint((low + high) // 2, high)
            else:
                shift_range = low

            if chosen_ids is None:
                chosen_ids = shift_range + torch.arange(group_ids[i])
            else:
                chosen_ids = torch.hstack((
                    shift_range + torch.arange(group_ids[i]),
                    chosen_ids
                ))

            high = shift_range - group_ids[i - 1]

        return seq[chosen_ids]

    @staticmethod
    def _random_group(n: int, max_seq_len: int):
        """
        Random group
        """
        subseq_len = []
        # group_ids = np.zeros(shape=n)

        while n > 0:
            if n == 1:
                subseq_len.append(1)
                break
            else:
                seq_len = np.random.randint(low=0, high=min(n, max_seq_len), size=1)[0] + 1
                n -= seq_len

                subseq_len.append(seq_len)

        # start = subseq_len[0]
        # for i in range(1, len(subseq_len)):
        #     end = subseq_len[i] + start

        #     group_ids[start : end] = i

        #     start = end

        # return group_ids

        return subseq_len

    def clean(self):
        shutil.rmtree(self.extract_dir)


if __name__ == "__main__":
    # out = ImputationDataset._random_choice(
    #     torch.arange(100),
    #     50,
    #     max_seq_len=20
    # )

    # print(out)
    data, labels = ImputationDataset(
        max_seq_len=1000,
        extract_dir="./data",
        archive_path="private-data.zip"
    ).get_dataset(mask_ratio=0.4)

    filled_data = idw_imputation(data, dist_type="haversine", n_contributor=5)
    filled_data = spline_imputation(data)
    # filled_data = median_imputation(data)

    # evaluate
    metrics_fn = MultiMetrics()
    metrics = {}

    for st_name, station in filled_data.items():
        ids, true_seq = labels[st_name]

        pred_seq = station["data"]["PM2.5"].values[ids]
        
        out = metrics_fn(
            torch.from_numpy(pred_seq),
            torch.from_numpy(true_seq)
        )

        for m in out:
            if m in metrics:
                metrics[m].append(out[m])
            else:
                metrics[m] = [out[m]]

    for m in metrics:
        metrics[m] = sum(metrics[m]) / len(metrics[m])

    print(metrics)