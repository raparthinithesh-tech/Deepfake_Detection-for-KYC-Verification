import torch
import torch.nn as nn
from torchvision import models

class ImageDeepfakeModel(nn.Module):
    def __init__(self):
        super(ImageDeepfakeModel, self).__init__()
        # Architecture explicitly requested by User's Kaggle Jupyter Notebook
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2) # Output layer for 2 classes (0: Real, 1: Fake)
        
    def forward(self, x):
        return self.model(x)

class VideoDeepfakeModel(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=1):
        super(VideoDeepfakeModel, self).__init__()
        
        # Video Frame Extractor matching Image ResNet18 backbone
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Strip final FC layer to get raw features (size 512)
        self.encoder.fc = nn.Identity()
        
        # LSTM for temporal dynamics analysis (sequence of frames)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        # x shape: (Batch, Seq_Length, C, H, W)
        batch_size, seq_len, c, h, w = x.size()
        
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.encoder(x)
        
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        
        return out
