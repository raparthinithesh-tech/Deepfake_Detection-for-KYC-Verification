import streamlit as st
import tempfile
import os
import torch
from PIL import Image
from kyc_pipeline import KYCPipeline

st.set_page_config(page_title="Deepfake & KYC Dashboard", layout="wide")

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_pipeline():
    # Load pipeline without weights for demo purposes if weights don't exist
    return KYCPipeline(device=device)

try:
    kyc = load_pipeline()
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose an Operation:",
    ["Home", "Image Deepfake", "Video Deepfake", "KYC Verification"]
)

if option == "Home":
    st.title("Deepfake Detection & KYC Verification")
    st.markdown("""
    Welcome to the Deepfake Detection and KYC Verification Dashboard!
    
    Please use the sidebar to navigate between:
    - **Image Deepfake**: Upload a single image to detect if it is a deepfake or real.
    - **Video Deepfake**: Upload a video to detect deepfakes using temporal frame analysis.
    - **KYC Verification**: An end-to-end KYC process that matches the face on an **Aadhaar** or **PAN Card** against a live selfie, while simultaneously running deepfake detection on the selfie to prevent spoofing.
    """)

elif option == "Image Deepfake":
    st.title("Image Deepfake Detection")
    st.markdown("Upload an image to verify its authenticity.")
    
    img_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if img_file:
        image = Image.open(img_file).convert('RGB')
        st.image(image, caption="Uploaded Image", width=400)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing artifacts for deepfakes..."):
                status, prob = kyc.detect_deepfake_image(image)
                if status == 'Real':
                    st.success(f"Result: {status} (Fake Probability: {prob*100:.2f}%)")
                else:
                    st.error(f"Result: {status} (Fake Probability: {prob*100:.2f}%)")

elif option == "Video Deepfake":
    st.title("Video Deepfake Detection")
    st.markdown("Upload a video to analyze frames for temporal deepfake inconsistencies.")
    
    vid_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    if vid_file:
        st.video(vid_file)
        
        if st.button("Analyze Video"):
            with st.spinner("Extracting frames and analyzing..."):
                file_ext = vid_file.name.split('.')[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_vid:
                    tmp_vid.write(vid_file.getvalue())
                    tmp_vid_path = tmp_vid.name
                
                try:
                    status, prob = kyc.detect_deepfake_video(tmp_vid_path)
                    if status == 'Real':
                        st.success(f"Result: {status} (Fake Probability: {prob*100:.2f}%)")
                    else:
                        st.error(f"Result: {status} (Fake Probability: {prob*100:.2f}%)")
                finally:
                    os.remove(tmp_vid_path)

elif option == "KYC Verification":
    st.title("KYC Verification (Aadhaar & PAN)")
    st.markdown("""
    Upload your **Aadhaar/PAN card** along with a **live selfie** (Image or Video) to verify your identity. 
    The system extracts the face from the ID card and matches it against the live selfie. Ensure the live selfie isn't a spoofed deepfake.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Official ID Document")
        id_card_file = st.file_uploader("Upload Aadhaar / PAN Card", type=['jpg', 'jpeg', 'png'], key="id_card")
        if id_card_file:
            id_image = Image.open(id_card_file)
            st.image(id_image, caption="Uploaded ID Card", use_container_width=True)

    with col2:
        st.subheader("2. Live Capture")
        live_media_file = st.file_uploader("Upload Live Selfie (Image/Video)", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'], key="live_media")
        
        live_type = None
        if live_media_file:
            file_ext = live_media_file.name.split('.')[-1].lower()
            if file_ext in ['jpg', 'jpeg', 'png']:
                live_image = Image.open(live_media_file)
                st.image(live_image, caption="Live Selfie", use_container_width=True)
                live_type = 'image'
            else:
                st.video(live_media_file)
                live_type = 'video'

    st.divider()
    
    if st.button("Run KYC Face Match & Liveness Check", use_container_width=True):
        if not id_card_file or not live_media_file:
            st.warning("Please upload both the ID Document AND the Live Media to proceed.")
        else:
            with st.spinner("Processing KYC Data..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_id:
                    tmp_id.write(id_card_file.getvalue())
                    tmp_id_path = tmp_id.name

                with tempfile.NamedTemporaryFile(delete=False, suffix="."+file_ext) as tmp_live:
                    tmp_live.write(live_media_file.getvalue())
                    tmp_live_path = tmp_live.name

                try:
                    if live_type == 'image':
                        result_msg = kyc.verify_identity(id_card_path=tmp_id_path, live_image_path=tmp_live_path)
                    else:
                        result_msg = kyc.verify_identity(id_card_path=tmp_id_path, live_video_path=tmp_live_path)

                    if "Passed" in result_msg:
                        st.success(result_msg)
                        st.balloons()
                    else:
                        st.error(result_msg)
                except Exception as e:
                    st.error(f"Error during verification: {e}")
                finally:
                    os.remove(tmp_id_path)
                    os.remove(tmp_live_path)
