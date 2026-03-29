import torch
import cv2
import os
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from models import ImageDeepfakeModel, VideoDeepfakeModel

class KYCPipeline:
    def __init__(self, df_image_model_path=None, df_video_model_path=None, device='cuda'):
        self.device = torch.device(device)
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Exact transforms from User's Jupyter Notebook
        self.df_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load Custom ResNet18 Model matching Jupyter Notebook
        self.image_df_model = ImageDeepfakeModel().to(self.device)
        if df_image_model_path and os.path.exists(df_image_model_path):
            print(f"Loading custom weights from {df_image_model_path}!")
            state_dict = torch.load(df_image_model_path, map_location=self.device)
            
            # The Kaggle notebook saved the weights directly from the ResNet18 instance.
            # Our custom ImageDeepfakeModel class wraps ResNet18 inside 'self.model'.
            # Thus, we load the state_dict explicitly into self.image_df_model.model !
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
                
            try:
                self.image_df_model.model.load_state_dict(state_dict)
            except RuntimeError:
                # Fallback if the user actually trained the wrapped ImageDeepfakeModel via train.py
                self.image_df_model.load_state_dict(state_dict, strict=False)
        else:
            print("WARNING: Custom weights not found. Dashboard will run with randomly initialized ResNet18 until you provide the .pth file!")
        self.image_df_model.eval()

        self.video_df_model = VideoDeepfakeModel().to(self.device)
        if df_video_model_path and os.path.exists(df_video_model_path):
            state_dict = torch.load(df_video_model_path, map_location=self.device)
            self.video_df_model.load_state_dict(state_dict, strict=False)
        self.video_df_model.eval()

    def detect_deepfake_image(self, image_pil):
        # The user's notebook processes the ENTIRE RAW IMAGE using ResNet18 without MTCNN face-cropping!
        input_tensor = self.df_transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.image_df_model(input_tensor)
            # Exactly matching Notebook Logits-Softmax logic
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

            label_map = {0: "Real", 1: "Fake"}
            predicted_label = label_map[pred]
            confidence = probs[0][pred].item()

        return predicted_label, confidence

    def detect_deepfake_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(10): # Uniform 10 frames
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            frames.append(self.df_transform(pil_img))
        cap.release()
        
        if len(frames) == 0:
            return 'Error', 0.0

        seq = torch.stack(frames).unsqueeze(0).to(self.device) # [1, 10, C, H, W]
        with torch.no_grad():
            output = self.video_df_model(seq)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            
            label_map = {0: "Real", 1: "Fake"}
            predicted_label = label_map[pred]
            confidence = probs[0][pred].item()
            
        return predicted_label, confidence

    def extract_face_embeddings(self, image_pil):
        # MTCNN is strictly used ONLY for ID Verification embeddings now
        img_cropped = self.mtcnn(image_pil)
        if img_cropped is not None:
            img_cropped = img_cropped.unsqueeze(0).to(self.device)
            embeddings = self.resnet(img_cropped)
            return embeddings.detach()
        return None

    def verify_identity(self, id_card_path, live_image_path=None, live_video_path=None):
        print("--- Starting KYC Verification ---")
        id_image = Image.open(id_card_path).convert('RGB')
        
        print("Extracting Face from ID Card...")
        id_embedding = self.extract_face_embeddings(id_image)
        if id_embedding is None:
            return "Failed to detect face in the ID card."

        if live_image_path:
            live_image = Image.open(live_image_path).convert('RGB')
            print("Running Deepfake Analysis on Live Image...")
            df_status, p = self.detect_deepfake_image(live_image)
            
            if df_status == 'Fake':
                return "KYC Failed! Live image is a deepfake."

            print("Checking Face Match...")
            live_embedding = self.extract_face_embeddings(live_image)
            
        elif live_video_path:
            print("Running Deepfake Analysis on Live Video...")
            df_status, p = self.detect_deepfake_video(live_video_path)

            if df_status == 'Fake':
                return "KYC Failed! Live video is a deepfake sequence."
            
            cap = cv2.VideoCapture(live_video_path)
            ret, frame = cap.read()
            cap.release()
            live_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            print("Checking Face Match...")
            live_embedding = self.extract_face_embeddings(live_image)

        if live_embedding is None:
            return "Failed to detect face in live media."

        dist = torch.dist(id_embedding, live_embedding).item()
        is_match = dist < 1.0 # threshold
        
        if is_match:
            return f"KYC Passed! Identity matched. (Distance: {dist:.4f})"
        else:
            return f"KYC Failed. Face does not match ID Card. (Distance: {dist:.4f})"
