import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from pathlib import Path

# 1. Page Configuration & Styling
st.set_page_config(page_title="Cocoa Disease Detector", page_icon="🌱", layout="centered")

st.markdown('<div style="font-size:40px; font-weight:bold; color: #2E7D32; text-align: center;">🌱 Cocoa Leaf Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:18px; text-align: center; margin-bottom: 30px; color: #555555;">Powered by YOLOv8 — Deep Learning Academic Defense Prototype</div>', unsafe_allow_html=True)

st.divider()

# 2. File Routing
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    return None

with st.spinner("Initializing neural network brains..."):
    model = load_model()

# 3. Sidebar Project Meta-Information (Cleaned layout parsing)
st.sidebar.header("📋 Project Specifications")
st.sidebar.markdown("**Author:** NGAH")
st.sidebar.markdown("**Model Architecture:** YOLOv8n")
st.sidebar.markdown("**Dataset Scale:** 3,870 Images")
st.sidebar.markdown("**Training Hardware:** Local CPU Intel Pentium")
st.sidebar.markdown("**Target Classes:** Healthy, CSSVD, Anthracnose")

# 4. Drag and Drop User Interface
uploaded_file = st.file_uploader("Upload a clear image of a cocoa leaf (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Input Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("⚡ AI Diagnostic")
        
        if model is None:
            st.error("Error: Custom weights file 'best.pt' not found.")
        else:
            with st.spinner("Scanning leaf pixels..."):
                temp_path = os.path.join(BASE_DIR, "temp_leaf_upload.jpg")
                image.save(temp_path)
                
                results = model.predict(source=temp_path, imgsz=256, conf=0.05, iou=0.4)
                
                res_plotted = results[0].plot()
                predicted_image = Image.fromarray(res_plotted[:, :, ::-1])
                
                st.image(predicted_image, use_container_width=True)
                
                if len(results[0].boxes) == 0:
                    st.warning("No anomalies identified with high confidence.")
                else:
                    st.success(f"Detected {len(results[0].boxes)} diagnostic marker(s)!")
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)

st.divider()
st.caption("Developed for Academic Thesis Evaluation — Running on Cloud Production Mirror.")
