import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2
# Load mô hình đã lưu
path ='D:\DATA\slidenfile\itcv\model.pkl'
with open(path, 'rb') as f:
    loaded_model = pickle.load(f)
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Tạo giao diện người dùng
st.title("Nhận diện nấm không độc hay có độc")
uploaded_image = st.file_uploader("Tải lên ảnh", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    # Trích xuất histogram màu
    st.image(image, channels="BGR")
    hist = extract_color_histogram(image, bins=(8, 8, 8))
    X =[]
    X.append(hist)
    
    prediction = loaded_model.predict(X)
    if prediction[0] == 1:
        st.write("Dự đoán: NẤM KHÔNG ĐỘC")
    else:
        st.write("Dự đoán: NẤM ĐỘC")
