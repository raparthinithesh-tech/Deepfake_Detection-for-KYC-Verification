import torch
import torch.nn as nn
import torch.optim as optim
from models import ImageDeepfakeModel, VideoDeepfakeModel
from data_loader import get_image_dataloaders, get_video_dataloaders
from tqdm import tqdm
import os
import argparse

def train_model(model, train_loader, val_loader, num_epochs=25, device='cuda', save_path='model.pth'):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    # User explicitly requested the Adam Optimizer
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
        
        # Testing Phase
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
        print(f"Epoch {epoch+1}: Train Loss={train_loss/total_train:.4f}, Train Acc={train_acc:.4f} | Test Loss={val_loss/total_val:.4f}, Test Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model to {save_path} (Test Acc: {best_acc:.4f})")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, choices=['image', 'video'], required=True, help="Train on Images or Videos")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to kaggle data (containing Real and Fake subfolders)")
    parser.add_argument('--split_ratio', type=float, default=0.8, help="Ratio to split dataset into training vs testing (e.g. 0.8 for 80/20)")
    parser.add_argument('--epochs', type=int, default=25, help="Number of epochs to train (requested >20)")
    args = parser.parse_args()

    # Automatically split Kaggle folders into PyTorch Training and Testing sets
    if args.dataset_type == 'image':
        print(f"Loading Image Dataset and applying {args.split_ratio*100:.0f}% Train / {(1-args.split_ratio)*100:.0f}% Test split...")
        train_loader, test_loader = get_image_dataloaders(args.data_dir, batch_size=32, split_ratio=args.split_ratio)
        model = ImageDeepfakeModel(model_name='efficientnet_b4')
        train_model(model, train_loader, test_loader, num_epochs=args.epochs, device=device, save_path='deepfake_image_model.pth')
    
    else:
        print(f"Loading Video Dataset and applying {args.split_ratio*100:.0f}% Train / {(1-args.split_ratio)*100:.0f}% Test split...")
        train_loader, test_loader = get_video_dataloaders(args.data_dir, seq_length=10, batch_size=8, split_ratio=args.split_ratio)
        model = VideoDeepfakeModel(model_name='efficientnet_b0')
        train_model(model, train_loader, test_loader, num_epochs=args.epochs, device=device, save_path='deepfake_video_model.pth')
