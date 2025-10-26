# ============================================================
# 🧠 AI Vision Dashboard — Analitik Lengkap (TFLite + YOLO)
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

# CSS untuk tampilan analitik modern
st.markdown("""
    <style>
        body { background-color: #f8fafc; font-family: 'Segoe UI', sans-serif; }
        .stApp { background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%); }
        h1, h2, h3 { color: #1e293b; }
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
        .stButton button:hover { background-color: #1d4ed8; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI Vision Dashboard")
st.markdown("### Analisis Data Gambar dengan *Klasifikasi* dan *Deteksi Objek*")

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
# 📊 Fungsi Dataset Info
# ============================================================
def get_dataset_info(dataset_path):
    classes, counts, avg_sizes = [], [], []
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.endswith((".jpg", ".jpeg", ".png"))]
            sizes = [os.path.getsize(os.path.join(class_path, img)) / 1024 for img in images]
            classes.append(class_name)
            counts.append(len(images))
            avg_sizes.append(np.mean(sizes) if sizes else 0)
    return pd.DataFrame({
        "Kelas": classes,
        "Jumlah Gambar": counts,
        "Rata-rata Ukuran (KB)": avg_sizes
    })

# ============================================================
# 🧭 Sidebar Menu
# ============================================================
menu = st.sidebar.radio(
    "📌 Pilih Mode:",
    ["🏞️ Visualisasi Dataset", "📷 Klasifikasi Gambar", "🎯 Deteksi Objek (YOLO)", "ℹ️ Tentang Aplikasi"]
)

# ============================================================
# 🏞️ VISUALISASI DATASET
# ============================================================
if menu == "🏞️ Visualisasi Dataset":
    st.header("📊 Visualisasi Dataset")

    dataset_type = st.selectbox("Pilih Jenis Dataset:", ["Klasifikasi Gambar", "Deteksi Objek"])
    dataset_path = "data/Klasifikasi Gambar" if dataset_type == "Klasifikasi Gambar" else "data/Object Detection"

    if not os.path.exists(dataset_path):
        st.error(f"❌ Folder '{dataset_path}' tidak ditemukan.")
    else:
        df_info = get_dataset_info(dataset_path)
        st.dataframe(df_info, use_container_width=True)

        # Grafik jumlah gambar per kelas
        fig1 = px.bar(
            df_info, x="Kelas", y="Jumlah Gambar",
            color="Kelas",
            title="📸 Jumlah Gambar per Kelas",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

        # Grafik rata-rata ukuran file
        fig2 = px.bar(
            df_info, x="Kelas", y="Rata-rata Ukuran (KB)",
            color="Kelas",
            title="💾 Rata-rata Ukuran File per Kelas",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # Contoh gambar
        st.subheader("🖼️ Contoh Gambar Tiap Kelas")
        cols = st.columns(min(5, len(df_info)))
        for i, class_name in enumerate(df_info["Kelas"][:5]):
            class_path = os.path.join(dataset_path, class_name)
            img_files = [f for f in os.listdir(class_path) if f.endswith((".jpg", ".jpeg", ".png"))]
            if img_files:
                img = Image.open(os.path.join(class_path, img_files[0]))
                cols[i].image(img, caption=class_name, use_container_width=True)

# ============================================================
# 📷 KLASIFIKASI GAMBAR
# ============================================================
elif menu == "📷 Klasifikasi Gambar":
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

        # Fallback
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
            title="📈 Distribusi Probabilitas Prediksi",
            color=class_labels,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🎯 DETEKSI OBJEK YOLO
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

    Dashboard ini bersifat interaktif dan informatif, membantu dalam eksplorasi data gambar
    dengan pendekatan berbasis **AI Vision**.

    👩‍💻 **Dikembangkan oleh:** *Nayma Alaydia*  
    📘 **Mata Kuliah:** Praktikum Pemrograman Big Data  
    🏫 **Tujuan:** Demonstrasi implementasi *Big Data & Computer Vision*
    """)
