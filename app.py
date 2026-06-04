import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# 1. Page Configuration & Styling
st.set_page_config(page_title="Cocoa Disease Detector", page_icon="🌱", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size:40px; font-weight:bold; color: #2E7D32; text-align: center; }
    .subtitle { font-size:18px; text-align: center; margin-bottom: 30px; color: #555555; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">🌱 Cocoa Leaf Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by YOLOv8 — Deep Learning Academic Defense Prototype</div>', unsafe_allow_html=True)

st.divider()

# 2. Dynamic Path to your freshly trained weights (Relative path for GitHub Cloud)
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    """Loads the model once and caches it to protect laptop RAM."""
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    return None

with st.spinner("Initializing neural network brains..."):
    model = load_model()

# 3. Sidebar Project Meta-Information
st.sidebar.header("📋 Project Specifications")
st.sidebar.info("""
**Author:** NGAH  
**Model Architecture:** YOLOv8n  
**Dataset Scale:** 3,870 Images  
**Training Hardware:** Local CPU Intel Pentium  
**Target Classes:** Healthy, CSSVD, Anthracnose
""")

# 4. Drag and Drop User Interface
uploaded_file = st.file_uploader("Upload a clear image of a cocoa leaf (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display input image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Input Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("⚡ AI Diagnostic")
        if model is None:
            st.error("Error: Trained model file 'best.pt' not found. Please verify your file path.")
        else:
            with st.spinner("Scanning leaf pixels..."):
                # Save temp file for YOLO disk pipeline compatibility
                temp_path = "temp_leaf_upload.jpg"
                image.save(temp_path)
                
                # Inference execution using matching training image size and synchronized confidence thresholds
                results = model.predict(source=temp_path, imgsz=256, conf=0.15, iou=0.4)
                
                # Render predicted bounding boxes
                res_plotted = results[0].plot()
                predicted_image = Image.fromarray(res_plotted[:, :, ::-1]) # Convert BGR back to RGB
                
                st.image(predicted_image, use_container_width=True)
                
                # Check if any bounding boxes were actually found
                if len(results[0].boxes) == 0:
                    st.warning("No anomalies identified with high confidence.")
                else:
                    st.success(f"Detected {len(results[0].boxes)} diagnostic marker(s)!")
                
                # Remove temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

st.divider()
st.caption("Developed for Academic Thesis Evaluation — Running on Localhost Mirror.")
