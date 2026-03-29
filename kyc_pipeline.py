import torch
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from transformers import pipeline
from models import ImageDeepfakeModel, VideoDeepfakeModel

class KYCPipeline:
    def __init__(self, df_image_model_path=None, df_video_model_path=None, device='cuda'):
        self.device = torch.device(device)
        self.device_id = 0 if self.device.type == 'cuda' else -1
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Transforms for Custom Trained Deepfake Models
        self.df_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.use_hf_for_df = False
        
        # Load Pretrained Advanced Deepfake Model from HuggingFace if path NOT provided!
        if not df_image_model_path and not df_video_model_path:
            print("Loading HuggingFace State-Of-The-Art Deepfake Model (dima806)...")
            # Using a Vision Transformer fine-tuned natively on deepfakes!
            self.hf_df_pipeline = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection", device=self.device_id)
            self.use_hf_for_df = True
        else:
            # Load your Custom Models if you actually trained them with the script
            # Load DF Image Model
            self.image_df_model = ImageDeepfakeModel(model_name='efficientnet_b4').to(self.device)
            if df_image_model_path:
                self.image_df_model.load_state_dict(torch.load(df_image_model_path, map_location=self.device))
            self.image_df_model.eval()

            # Load DF Video Model
            self.video_df_model = VideoDeepfakeModel(model_name='efficientnet_b0').to(self.device)
            if df_video_model_path:
                self.video_df_model.load_state_dict(torch.load(df_video_model_path, map_location=self.device))
            self.video_df_model.eval()

    def detect_deepfake_image(self, image_pil):
        # image_pil: A PIL Image containing face
        if self.use_hf_for_df:
            # High-accuracy Deepfake Liveness networks are incredibly sensitive to digital composites.
            # Things like black structural borders, polaroid effects, scanned edges, and studio watermarks
            # will INSTANTLY be recognized as "synthetic overlays" and flagged 99% Fake.
            # We use an MTCNN smart-crop with a generous 1.5x margin to remove those borders and watermarks before analysis!
            smart_crop = image_pil
            boxes, probs = self.mtcnn.detect(image_pil)
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]
                w_face, h_face = box[2] - box[0], box[3] - box[1]
                margin_x, margin_y = w_face * 1.2, h_face * 1.5  # Large context margin stops scaling artifacts
                w, h = image_pil.size
                
                # Expand box and clamp to image bounds
                x1 = max(0, int(box[0] - margin_x))
                y1 = max(0, int(box[1] - margin_y))
                x2 = min(w, int(box[2] + margin_x))
                y2 = min(h, int(box[3] + margin_y))
                smart_crop = image_pil.crop((x1, y1, x2, y2))

            results = self.hf_df_pipeline(smart_crop)
            top_pred = results[0]
            label = top_pred['label'].lower()
            score = top_pred['score']
            
            status = 'Fake' if 'fake' in label else 'Real'
            fake_prob = score if status == 'Fake' else (1.0 - score)
            return status, fake_prob
        else:
            crop = image_pil
            boxes, probs = self.mtcnn.detect(image_pil)
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]
                # Expand box slightly for context
                w, h = image_pil.size
                x1, y1 = max(0, int(box[0])-30), max(0, int(box[1])-30)
                x2, y2 = min(w, int(box[2])+30), min(h, int(box[3])+30)
                crop = image_pil.crop((x1, y1, x2, y2))
                
            img_tensor = self.df_transform(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.image_df_model(img_tensor)
                prob = torch.sigmoid(output).item()
            
            return 'Fake' if prob > 0.5 else 'Real', prob

    def detect_deepfake_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        pil_frames = []
        for _ in range(10): # Uniform 10 frames
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            pil_frames.append(pil_img)
            frames.append(self.df_transform(pil_img))
        cap.release()
        
        if len(frames) == 0:
            return 'Error', 0.0

        if self.use_hf_for_df:
            # Run image pipeline frame-by-frame and average
            fake_probs = []
            for frame_img in pil_frames:
                status, prob = self.detect_deepfake_image(frame_img)
                fake_probs.append(prob)
            avg_prob = sum(fake_probs) / len(fake_probs)
            return 'Fake' if avg_prob > 0.5 else 'Real', avg_prob
        else:
            seq = torch.stack(frames).unsqueeze(0).to(self.device) # [1, 10, C, H, W]
            with torch.no_grad():
                output = self.video_df_model(seq)
                prob = torch.sigmoid(output).item()
            
            return 'Fake' if prob > 0.5 else 'Real', prob

    def extract_face_embeddings(self, image_pil):
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
                return f"KYC Failed! Live image is a deepfake. ({p*100:.2f}% Fake)"

            print("Checking Face Match...")
            live_embedding = self.extract_face_embeddings(live_image)
            
        elif live_video_path:
            print("Running Deepfake Analysis on Live Video...")
            df_status, p = self.detect_deepfake_video(live_video_path)

            if df_status == 'Fake':
                return f"KYC Failed! Live video is a deepfake sequence. ({p*100:.2f}% Fake)"
            
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

if __name__ == '__main__':
    print("KYC Pipeline configured with HuggingFace ViT for Deepfake Detection.")
