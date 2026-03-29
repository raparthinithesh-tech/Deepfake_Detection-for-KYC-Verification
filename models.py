import torch
import torch.nn as nn
import timm

class ImageDeepfakeModel(nn.Module):
    def __init__(self, model_name='efficientnet_b4', pretrained=True, num_classes=1):
        super(ImageDeepfakeModel, self).__init__()
        # Using a powerful vision model like EfficientNet for deepfake detection
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.fc = nn.Linear(self.encoder.num_features, num_classes)
        
    def forward(self, x):
        features = self.encoder(x)
        out = self.fc(features)
        return out


class VideoDeepfakeModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True, hidden_dim=256, num_layers=1, num_classes=1):
        super(VideoDeepfakeModel, self).__init__()
        
        # Frame Feature Extractor
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # LSTM for temporal dynamics analysis (sequence of frames)
        self.lstm = nn.LSTM(
            input_size=self.encoder.num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Seq_Length, C, H, W)
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape to process frames through encoder
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.encoder(x)
        
        # Reshape back to sequences
        features = features.view(batch_size, seq_len, -1)
        
        # Pass sequence to LSTM
        lstm_out, _ = self.lstm(features)
        
        # Take the output of the last sequential frame
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        
        return out

if __name__ == '__main__':
    # Test instantiating the models
    img_model = ImageDeepfakeModel()
    print("Image Model Created. Number of parameters:", sum(p.numel() for p in img_model.parameters()))

    vid_model = VideoDeepfakeModel()
    print("Video Model Created. Number of parameters:", sum(p.numel() for p in vid_model.parameters()))
