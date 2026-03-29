import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image

def get_image_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        # The parent subset tracks the original dataset, which has target_transform applied
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

class VideoDataset(Dataset):
    def __init__(self, data_dir, seq_length=10):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.videos = []
        self.labels = []
        
        if os.path.isdir(os.path.join(data_dir, 'Real')) and os.path.isdir(os.path.join(data_dir, 'Fake')):
            class_map = {'Fake': 1, 'Real': 0}
        else:
            class_map = {'fake': 1, 'real': 0}
            
        for cls_name, label in class_map.items():
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.exists(cls_dir):
                print(f"Directory missing: {cls_dir}")
                continue
                
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.videos.append(os.path.join(cls_dir, fname))
                    self.labels.append(label)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_cnt == 0:
             cap.release()
             return [Image.new('RGB', (224, 224)) for _ in range(self.seq_length)]

        indices = [int(i * frame_cnt / self.seq_length) for i in range(self.seq_length)]
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            frames.append(pil_img)
            
        cap.release()
        
        while len(frames) < self.seq_length:
            frames.append(frames[-1] if len(frames) > 0 else Image.new('RGB', (224, 224)))
            
        return frames[:self.seq_length]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid_path = self.videos[idx]
        label = self.labels[idx]
        return self.extract_frames(vid_path), label

class VideoDatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        frames, y = self.subset[index]
        if self.transform:
            frames = [self.transform(f) for f in frames]
        else:
            frames = [transforms.ToTensor()(f) for f in frames]
        frames_tensor = torch.stack(frames)
        return frames_tensor, torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.subset)

def get_image_dataloaders(data_dir, batch_size=32, split_ratio=0.8):
    # Load entire dataset
    dataset = datasets.ImageFolder(root=data_dir)
    
    # Map Fake -> 1, Real -> 0 standard for KYC
    if dataset.class_to_idx.get('Real') == 1 and dataset.class_to_idx.get('Fake') == 0:
        dataset.target_transform = lambda y: 1 - y
    elif dataset.class_to_idx.get('real') == 1 and dataset.class_to_idx.get('fake') == 0:
        dataset.target_transform = lambda y: 1 - y
        
    # Split training and testing automatically
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_subset, test_subset = random_split(dataset, [train_size, test_size])
    
    # Wrap subsets to apply individual augmentation correctly!
    train_ds = DatasetWrapper(train_subset, transform=get_image_transforms(train=True))
    test_ds = DatasetWrapper(test_subset, transform=get_image_transforms(train=False))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader

def get_video_dataloaders(data_dir, seq_length=10, batch_size=8, split_ratio=0.8):
    dataset = VideoDataset(data_dir, seq_length=seq_length)
    
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_subset, test_subset = random_split(dataset, [train_size, test_size])
    
    train_ds = VideoDatasetWrapper(train_subset, transform=get_image_transforms(train=True))
    test_ds = VideoDatasetWrapper(test_subset, transform=get_image_transforms(train=False))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader
