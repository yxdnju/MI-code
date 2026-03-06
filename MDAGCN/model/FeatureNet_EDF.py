import torch
import torch.nn as nn
import torch.nn.functional as F


class LeadAttention(nn.Module):
    """
    Lead Attention Module: Learn the importance of each channel (Lead) in the feature dimension.
    Input Shape: [Batch, Channels, Feature_Dim]
    """

    def __init__(self, channels, feature_dim):
        super(LeadAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: [B, C, D]
        weights = self.attention(x)  # [B, C, 1]
        out = x * weights
        return torch.sum(out, dim=1)  # [B, D]


class FeatureBlock(nn.Module):
    def __init__(self, kernel_s, stride_s, kernel_l, stride_l, out_channels=64):
        super(FeatureBlock, self).__init__()
        activation = nn.ReLU

        # Branch 1: Small Filter
        self.feature_small = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=kernel_s, stride=stride_s, padding=kernel_s // 2),
            nn.BatchNorm1d(32),
            activation(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout1d(0.2),
            nn.Conv1d(32, out_channels, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(out_channels),
            activation(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Branch 2: Large Filter
        self.feature_large = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=kernel_l, stride=stride_l, padding=kernel_l // 2),
            nn.BatchNorm1d(32),
            activation(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout1d(0.2),
            nn.Conv1d(32, out_channels, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(out_channels),
            activation(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        f_s = self.feature_small(x)
        f_l = self.feature_large(x)

        # Concatenate small and large filter features
        out = torch.cat([self.flatten(f_s), self.flatten(f_l)], dim=1)  # [B, 128]
        return out


class FeatureNet(nn.Module):
    def __init__(self, channels=4):
        super(FeatureNet, self).__init__()

        # Modal-specific Feature Streams
        self.eeg_stream = FeatureBlock(kernel_s=25, stride_s=3, kernel_l=200, stride_l=25)
        self.eog_stream = FeatureBlock(kernel_s=25, stride_s=3, kernel_l=200, stride_l=25)
        self.emg_stream = FeatureBlock(kernel_s=5, stride_s=1, kernel_l=50, stride_l=5)

        # Modal-specific Weights
        self.eeg_weight = nn.Parameter(torch.ones(1))
        self.eog_weight = nn.Parameter(torch.ones(1))
        self.emg_weight = nn.Parameter(torch.ones(1))

        self.feature_dim = 128

        self.lead_attention = LeadAttention(channels, self.feature_dim)

        self.norm = nn.LayerNorm(self.feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5)
        )

    def get_feature(self, x):
        """
        Extract features from each channel (Lead) using modal-specific feature streams.
        """
        B, C, T = x.shape
        features = []
        weights = torch.softmax(torch.stack([self.eeg_weight, self.eog_weight, self.emg_weight]), dim=0)

        for i in range(C):
            x_i = x[:, i, :].unsqueeze(1)

            # Modal classification
            if i <= 1:  # EEG (Fpz-Cz, Pz-Oz)
                f_out = self.eeg_stream(x_i) * weights[0]
            elif i == 2:  # EOG
                f_out = self.eog_stream(x_i) * weights[1]
            else:  # EMG
                f_out = self.emg_stream(x_i) * weights[2]

            features.append(f_out)

        # Stack features from all channels [B, C, D]
        features = torch.stack(features, dim=1)
        features = self.norm(features)
        return features

    def forward(self, x):
        # 1. Extract features from each channel
        features = self.get_feature(x)  # [B, 4, 128]

        # 2. Apply lead attention to fuse features from all channels
        merged_feature = self.lead_attention(features)  # [B, 128]

        # 3. Classify the fused feature
        logits = self.classifier(merged_feature)
        return logits