# Deepfake Detection for KYC Verification

This project contains code intended to detect deepfake images and videos, primarily for the application of a KYC (Know Your Customer) Verification pipeline.

We utilize the datasets provided by nithesh0402 on Kaggle, which typically follow an explicit directory format separating 'Real' and 'Fake' imagery.

## Setup Instructions

1. **Install requirements:**
   In your python environment, ensure you install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Formats**
   Your training data directories should be formatted like this:
   ```text
   dataset/
       train/
           Real/
               img1.jpg
               ...
           Fake/
               img2.jpg
               ...
       val/
           Real/
           Fake/
   ```

3. **Training the Image Deepfake Detector:**
   ```bash
   python train.py --dataset_type image --train_dir path/to/dataset/train --val_dir path/to/dataset/val --epochs 10
   ```
   
4. **Training the Video Deepfake Detector:**
   ```bash
   python train.py --dataset_type video --train_dir path/to/video/dataset/train --val_dir path/to/video/dataset/val --epochs 10
   ```

## KYC Verification Pipeline

The file `kyc_pipeline.py` integrates the trained deepfake detection model with an explicit Identity Verification flow using `facenet-pytorch`.

**Process:**
1. Employs MTCNN to locate and extract faces from ID cards and live webcam images/videos.
2. Cross-references the identity by comparing face embeddings from `InceptionResnetV1` (Facenet).
3. Simultaneously streams the imagery/video through the trained `EfficientNet` (Image Model) or `LSTM` (Video Model) to guarantee liveness and avoid synthetic generative spoofing (Deepfakes).

Simply run or import the `KYCPipeline` class from `kyc_pipeline.py` providing the `.pth` weights you trained.
