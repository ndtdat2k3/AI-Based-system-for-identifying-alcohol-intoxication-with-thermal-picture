# drunk_detection_webapp.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Config UI
st.set_page_config(page_title="Drunk Detection App", page_icon="🔥", layout="wide")
st.markdown("""
    <style>
    .main-header {text-align: center; font-size: 2.5em; color: #FF6B6B; margin-bottom: 20px;}
    .upload-area {background-color: #F0F2F6; padding: 20px; border-radius: 10px; text-align: center;}
    .result-box {background-color: #E8F5E8; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;}
    .error-box {background-color: #FFEBEE; padding: 15px; border-radius: 10px; border-left: 5px solid #F44336;}
    </style>
""", unsafe_allow_html=True)

# Load models (từ documents: YOLO best.pt, ResNet state_dict)
@st.cache_resource
def load_models():
    try:
        # Full path cho YOLO 
        yolo_path = r'C:\Users\Administrator\Downloads\Năm 5 kì 1\Thị giác máy tính - CV (môn chiều t6)\Midterm\retrain\best.pt'
        yolo_model = YOLO(yolo_path)
        
        # Full path cho ResNet (pth file)
        resnet_path = r'C:\Users\Administrator\Downloads\Năm 5 kì 1\Thị giác máy tính - CV (môn chiều t6)\Midterm\retrain\resnet50_classifier_v3_from_scratch.pth'
        resnet_model = models.resnet50(weights=None)
        
        # === CẤU TRÚC MODEL PHẢI GIỐNG HỆT FILE TRAINING ===
        num_features = resnet_model.fc.in_features
        resnet_model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 2) # 2 classes
        )

        resnet_model.load_state_dict(torch.load(resnet_path, map_location='cpu'))
        resnet_model.eval()
        
        class_names = ['Drunk', 'Sober']  # Từ label_map doc
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet_model.to(device)
        
        return yolo_model, resnet_model, class_names, device
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None, None, None, None

# Transform ResNet (chỉ resize/normalize, không augmentation)
# === ĐẢM BẢO TRANSFORM GIỐNG HỆT transform_val TRONG FILE TRAINING ===
transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)), # Đảm bảo resize về 224x224
    transforms.ToTensor(),   # Chuyển tensor [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize ImageNet
])

# Pipeline 
def run_pipeline(image, yolo_model, resnet_model, class_names, device):
    # YOLO predict (tự resize 640x640 từ doc)
    results = yolo_model.predict(image, imgsz=640, conf=0.25, save=False)
    
    # Chuyển image (PIL) sang array (OpenCV BGR)
    # Vì results[0].orig_img là từ ảnh PIL, nó có thể là RGB
    open_cv_image = np.array(image)
    # Chuyển RGB (PIL) sang BGR (OpenCV)
    if open_cv_image.shape[2] == 3: # Đảm bảo là ảnh màu
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    if not results[0].boxes or len(results[0].boxes) == 0:
        return None, "Không detect được bounding box!", None
    
    # Bbox từ YOLO (như draw_boxes doc)
    box = results[0].boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    
    # Crop từ orig_img (array BGR)
    cropped = open_cv_image[y1:y2, x1:x2]
    
    # Fix: Convert cropped BGR to RGB (PIL) để transform
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    
    # ResNet classify 
    cropped_pil = Image.fromarray(cropped_rgb)
    input_tensor = transform_resnet(cropped_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = resnet_model(input_tensor)
        pred_class = torch.argmax(output, 1).item()
        confidence = torch.softmax(output, 1).max().item()
    
    pred_label = class_names[pred_class]
    
    # Trả về ảnh gốc (PIL) để hiển thị bbox
    final_bbox_img = results[0].plot() # Đây là ảnh BGR
    final_bbox_img_rgb = cv2.cvtColor(final_bbox_img, cv2.COLOR_BGR2RGB) # Chuyển sang RGB

    return cropped_rgb, pred_label, confidence, final_bbox_img_rgb  # Trả cropped_rgb để display

# UI chính
st.markdown('<h1 class="main-header">🔥 Drunk Detection with Thermal Images</h1>', unsafe_allow_html=True)
st.markdown("Upload ảnh nhiệt → Detect bbox (YOLO) → Classify trạng thái (ResNet-50).")

# Load models
yolo_model, resnet_model, class_names, device = load_models()
if yolo_model is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📋 Hướng dẫn")
    st.write("- Upload ảnh (.jpg, .png).")
    st.write("- Nhấn **Result** để chạy.")

# Upload (nút Insert Picture)
uploaded_file = st.file_uploader("Insert Picture:", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB') # Đảm bảo ảnh là RGB
    st.image(image, caption="Ảnh input", use_column_width=True)
    
    # Nút Result
    if st.button("Result", type="primary"):
        with st.spinner("Chạy pipeline..."):
            cropped_rgb, pred_label, confidence, bbox_img_rgb = run_pipeline(image, yolo_model, resnet_model, class_names, device)
            
            if cropped_rgb is None:
                st.markdown(f'<div class="error-box">❌ {pred_label}</div>', unsafe_allow_html=True)
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Ảnh Crop từ BBox (Input ResNet)")
                    st.image(cropped_rgb, caption="Cropped BBox", use_column_width=True)  # Giờ dùng cropped_rgb
                
                with col2:
                    st.markdown(f'<div class="result-box">🎯 **Label: {pred_label}**<br>📊 Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)
                
                # Ảnh gốc với bbox (đã là RGB)
                st.markdown("### Ảnh gốc với Bounding Box (YOLO - Vị trí khuôn mặt)")
                st.image(bbox_img_rgb, caption="Ảnh với BBox", use_column_width=True)

else:
    st.info("👆 Upload ảnh để bắt đầu!")

st.markdown("---")
st.markdown("**Dự án: Drunk Detection | YOLOv8 + ResNet-50**")
