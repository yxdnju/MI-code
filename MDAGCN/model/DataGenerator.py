import numpy as np

class kFoldGenerator_train():
    '''
    Data Generator
    '''
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k

    # Initializate
    def __init__(self, x, y):
        if len(x) != len(y):
            assert False, 'Data generator: Length of x or y is not equal to k.'
        self.k = len(x)
        # Standardize each subject independently
        self.x_list = []
        for subject_data in x:
            normalized_data = np.zeros_like(subject_data)
            for channel in range(subject_data.shape[1]):
                channel_data = subject_data[:, channel, :]
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                if std > 0:
                    normalized_data[:, channel, :] = (channel_data - mean) / std
                else:
                    normalized_data[:, channel, :] = channel_data - mean
            self.x_list.append(normalized_data)

        self.y_list = y

    # Get i-th fold
    def getFold(self, i):
        isFirst = True
        for p in range(self.k):
            if p == i:
                val_data = self.x_list[p]
                val_targets = self.y_list[p]
                print('  Fold #', p, ': val')
            elif p != i:
                if isFirst:
                    train_data = self.x_list[p]
                    train_targets = self.y_list[p]
                    isFirst = False
                else:
                    train_data = np.concatenate((train_data, self.x_list[p]))
                    train_targets = np.concatenate((train_targets, self.y_list[p]))
                print('  Fold #', p, ': train')
        return train_data, np.argmax(train_targets,1), val_data, np.argmax(val_targets,1)
