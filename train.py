import torch
import torch.nn as nn
import torch.optim as optim
from models import ImageDeepfakeModel, VideoDeepfakeModel
from data_loader import get_image_dataloader, get_video_dataloader
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda', save_path='model.pth'):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1) # [B, 1]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) >= 0.5
            correct_train += preds.eq(labels).sum().item()
            total_train += labels.size(0)
            
            pbar.set_postfix({'Loss': loss.item()})

        train_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) >= 0.5
                correct_val += preds.eq(labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / (total_val + 1e-8)
        print(f"Epoch {epoch+1}: Train Loss={train_loss/total_train:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss/total_val:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model to {save_path}")

if __name__ == '__main__':
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # User needs to set their directories
    # For example: data_dir = 'dataset/deepfake-and-real-images/train'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, choices=['image', 'video'], required=True, help="Train on Images or Videos")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to training data (containing Real/Fake subfolders)")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to validation data")
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    if args.dataset_type == 'image':
        train_loader = get_image_dataloader(args.train_dir, batch_size=32, train=True)
        val_loader = get_image_dataloader(args.val_dir, batch_size=32, train=False)
        model = ImageDeepfakeModel(model_name='efficientnet_b4')
        train_model(model, train_loader, val_loader, num_epochs=args.epochs, device=device, save_path='deepfake_image_model.pth')
    else:
        train_loader = get_video_dataloader(args.train_dir, batch_size=8, train=True)
        val_loader = get_video_dataloader(args.val_dir, batch_size=8, train=False)
        model = VideoDeepfakeModel(model_name='efficientnet_b0')
        train_model(model, train_loader, val_loader, num_epochs=args.epochs, device=device, save_path='deepfake_video_model.pth')
