# ============================================================
# 🧠 AI Vision Dashboard — Dual Model (TFLite + YOLOv8)
# ============================================================

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import os
import plotly.express as px

# ============================================================
# ⚙️ Konfigurasi Halaman
# ============================================================
st.set_page_config(
    page_title="AI Vision Dashboard",
    page_icon="🧠",
    layout="wide",
)

# Gaya CSS modern analitik
st.markdown("""
    <style>
        body {
            background-color: #f8fafc;
            font-family: 'Segoe UI', sans-serif;
        }
        .stApp {
            background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
        }
        h1, h2, h3 {
            color: #1e293b;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e40af 0%, #3b82f6 100%);
            color: white;
        }
        .stButton button {
            background-color: #2563eb;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5em 1em;
        }
        .stButton button:hover {
            background-color: #1d4ed8;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI Vision Dashboard")
st.markdown("### Analisis Data Gambar dengan *Klasifikasi* & *Deteksi Objek*")

# ============================================================
# 📦 Load Model
# ============================================================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Nayma Alaydia_Laporan 4.pt")

    interpreter = tf.lite.Interpreter(model_path="model/model_kecil.tflite")
    interpreter.allocate_tensors()

    return yolo_model, interpreter

yolo_model, classifier = load_models()

# ============================================================
# 🧭 Sidebar Menu
# ============================================================
menu = st.sidebar.radio(
    "📌 Pilih Mode:",
    ["📷 Klasifikasi Gambar", "🎯 Deteksi Objek (YOLO)", "ℹ️ Tentang Aplikasi"]
)

# ============================================================
# 📷 KLASIFIKASI GAMBAR
# ============================================================
if menu == "📷 Klasifikasi Gambar":
    st.header("📷 Klasifikasi Gambar Menggunakan TFLite")

    uploaded_file = st.file_uploader("Unggah gambar:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang diunggah", use_container_width=True)

        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        with st.spinner("🔮 Menganalisis gambar..."):
            input_details = classifier.get_input_details()
            output_details = classifier.get_output_details()

            classifier.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
            classifier.invoke()

            prediction = classifier.get_tensor(output_details[0]['index'])[0]
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        data_path = "data/Klasifikasi Gambar"
        class_labels = sorted(os.listdir(data_path))
        predicted_label = class_labels[class_index] if class_index < len(class_labels) else f"Kelas {class_index}"

        # Panel hasil
        col1, col2 = st.columns(2)
        col1.metric("🧩 Prediksi", predicted_label)
        col2.metric("🔢 Probabilitas", f"{confidence*100:.2f}%")

        # Logika fallback
        if confidence < 0.5:
            st.warning("⚠️ Model tidak yakin — gambar mungkin di luar domain model klasifikasi.")
        elif confidence > 0.8:
            st.success("✅ Model sangat yakin dengan hasil ini!")
        else:
            st.info("ℹ️ Model cukup yakin, tapi perlu verifikasi manual.")

        # Grafik probabilitas
        fig = px.bar(
            x=class_labels,
            y=prediction,
            title="📊 Distribusi Probabilitas Prediksi",
            color=class_labels,
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🎯 DETEKSI OBJEK (YOLO)
# ============================================================
elif menu == "🎯 Deteksi Objek (YOLO)":
    st.header("🎯 Deteksi Objek Menggunakan YOLOv8")

    uploaded_file = st.file_uploader("Unggah gambar:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang diunggah", use_container_width=True)

        with st.spinner("🚀 Mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()

        st.image(result_img, caption="📍 Hasil Deteksi", use_container_width=True)

        boxes = results[0].boxes
        if len(boxes) > 0:
            det_data = {
                "Label": [yolo_model.names[int(cls)] for cls in boxes.cls],
                "Confidence": [float(conf) for conf in boxes.conf],
                "Koordinat (x1,y1,x2,y2)": [box.xyxy.tolist()[0] for box in boxes]
            }
            df_det = pd.DataFrame(det_data)
            st.dataframe(df_det)
        else:
            st.warning("⚠️ Tidak ada objek yang terdeteksi.")

# ============================================================
# ℹ️ TENTANG APLIKASI
# ============================================================
elif menu == "ℹ️ Tentang Aplikasi":
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    ### 🎓 Dashboard UTS Praktikum Pemrograman Big Data  
    Dashboard ini dikembangkan untuk mendemonstrasikan dua model analisis data gambar:

    - 📷 *Klasifikasi Gambar* menggunakan model **TensorFlow Lite (CNN)**
    - 🎯 *Deteksi Objek* menggunakan model **YOLOv8 (.pt)**

    Dashboard ini dirancang agar bersifat interaktif dan informatif,  
    memungkinkan pengguna untuk mengeksplorasi data gambar dengan pendekatan berbasis **AI Vision**.
    
    👩‍💻 **Dikembangkan oleh:** *Nayma Alaydia*  
    📘 **Mata Kuliah:** Praktikum Pemrograman Big Data  
    🏫 **Tujuan:** Demonstrasi implementasi *Big Data & Computer Vision*
    """)

