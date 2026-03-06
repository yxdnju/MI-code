import numpy as np


class SleepEDFGenerator():
    '''
    Data generator for Sleep-EDF-78 dataset.
    '''

    def __init__(self, x_list, y_list, k=10):
        """
        x_list: include 153 subjects' data, each with shape [N, Channels, 3000]
        y_list: include 153 subjects' labels
        k: target fold number, default 10
        """
        if len(x_list) != len(y_list):
            raise ValueError("Data and Labels must have the same number of subjects.")

        self.target_k = k
        self.num_subjects = len(x_list)

        # 1. Standardize each subject independently
        print("Standardizing subjects individually...")
        self.processed_x = []
        for subject_data in x_list:
            normalized_data = np.zeros_like(subject_data)
            for channel in range(subject_data.shape[1]):
                ch_data = subject_data[:, channel, :]
                mean, std = np.mean(ch_data), np.std(ch_data)
                normalized_data[:, channel, :] = (ch_data - mean) / (std + 1e-8)
            self.processed_x.append(normalized_data)
        self.processed_y = y_list

        # 2. Calculate which subjects belong to each fold
        self.fold_map = self._get_fold_map()

    def _get_fold_map(self):
        """
        Assign each subject to a fold.
        """
        indices = np.arange(self.num_subjects)
        # np.random.shuffle(indices)

        fold_map = {}
        fold_size = self.num_subjects // self.target_k

        for i in range(self.target_k):
            start = i * fold_size
            end = (i + 1) * fold_size if i != self.target_k - 1 else self.num_subjects
            fold_map[i] = indices[start:end]
        return fold_map

    def getFold(self, i):
        """
        Get the i-th fold data.
        Train: include all subjects except the i-th fold.
        Val: include the i-th fold.
        """
        if i >= self.target_k:
            raise ValueError(f"Fold index {i} out of range {self.target_k}")

        print(f"Generating Fold #{i} (Merging subjects for 10-fold CV)...")

        val_indices = self.fold_map[i]
        train_indices = [idx for k, v in self.fold_map.items() if k != i for idx in v]

        val_data = np.concatenate([self.processed_x[idx] for idx in val_indices], axis=0)
        val_targets = np.concatenate([self.processed_y[idx] for idx in val_indices], axis=0)

        train_data = np.concatenate([self.processed_x[idx] for idx in train_indices], axis=0)
        train_targets = np.concatenate([self.processed_y[idx] for idx in train_indices], axis=0)

        print(f"  Train: {len(train_indices)} subjects, {train_data.shape[0]} samples")
        print(f"  Val:   {len(val_indices)} subjects, {val_data.shape[0]} samples")

        return train_data, train_targets, val_data, val_targets