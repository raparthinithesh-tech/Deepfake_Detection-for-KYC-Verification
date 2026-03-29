import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
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

class VideoDataset(Dataset):
    def __init__(self, data_dir, seq_length=10, transform=None):
        """
        data_dir: should contain subfolders 'Real' and 'Fake' with videos.
        seq_length: Number of frames to extract uniformly from a video.
        """
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.transform = transform
        
        self.videos = []
        self.labels = []
        
        # Expected Kaggle format usually places videos inside Real and Fake dirs
        if os.path.isdir(os.path.join(data_dir, 'Real')) and os.path.isdir(os.path.join(data_dir, 'Fake')):
            class_map = {'Fake': 1, 'Real': 0}
        else:
            # Maybe inside lowercase or with train/test splits inside, adjust as needed
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
             # Return empty frames fallback
             return [torch.zeros(3, 224, 224) for _ in range(self.seq_length)]

        indices = [int(i * frame_cnt / self.seq_length) for i in range(self.seq_length)]
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV) to RGB (PIL/Torch)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            
            if self.transform:
                frame_tensor = self.transform(pil_img)
            else:
                frame_tensor = transforms.ToTensor()(pil_img)
            frames.append(frame_tensor)
            
        cap.release()
        
        # Padding in case video is too short to extract full seq_length frames
        while len(frames) < self.seq_length:
            frames.append(torch.zeros_like(frames[0]) if len(frames) > 0 else torch.zeros(3, 224, 224))
            
        return torch.stack(frames[:self.seq_length]) # (seq_len, C, H, W)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid_path = self.videos[idx]
        label = self.labels[idx]
        
        frames = self.extract_frames(vid_path)
        return frames, torch.tensor(label, dtype=torch.float32)

def get_image_dataloader(data_dir, batch_size=32, train=True):
    # For Kaggle Image datasets containing "Real" and "Fake" folders
    transform = get_image_transforms(train)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    # Ensure Real is 0 and Fake is 1 (Usually datasets.ImageFolder does alphabetical: Fake=0, Real=1)
    # So we might need to map them manually. By default: 0 -> Fake, 1 -> Real. Let's fix it for KYC standard:
    # We want Fake=1, Real=0. We'll handle this mapping correctly here.
    if dataset.class_to_idx.get('Real') == 1 and dataset.class_to_idx.get('Fake') == 0:
        dataset.target_transform = lambda y: 1 - y # Reverse 0 to 1, and 1 to 0
    elif dataset.class_to_idx.get('real') == 1 and dataset.class_to_idx.get('fake') == 0:
        dataset.target_transform = lambda y: 1 - y
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)
    return loader

def get_video_dataloader(data_dir, seq_length=10, batch_size=8, train=True):
    transform = get_image_transforms(train)
    dataset = VideoDataset(data_dir, seq_length=seq_length, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)
    return loader
