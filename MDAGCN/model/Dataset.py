from torch.utils.data import Dataset

## Define the dataset class
# SimpleDataset: x, y

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y), \
            'The number of inputs(%d) and targets(%d) does not match.' % (len(x), len(y))
        self.x = x
        self.y = y
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y[index]
