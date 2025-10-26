# ============================================================
# 🧠 AI Vision Dashboard — Modern Streamlit Version
# ============================================================

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# ============================================================
# ⚙️ Konfigurasi Halaman
# ============================================================
st.set_page_config(
    page_title="Dashboard Nayma Alaydia",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* Gaya Umum */
        body {
            background-color: #f8fafc;
            font-family: 'Segoe UI', sans-serif;
        }
        .stApp {
            background: linear-gradient(180deg, #e0f2fe 0%, #ffffff 100%);
        }
        h1, h2, h3 {
            color: #0f172a;
        }
        .stProgress > div > div {
            background-color: #3b82f6;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 🔍 Fungsi Load Model
# ============================================================
@st.cache_resource
def load_models():
    # Load model YOLO
    yolo_model = YOLO("model/Nayma Alaydia_Laporan 4.pt")

    # Load model TensorFlow Lite
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path="model/model_kecil.tflite")
    interpreter.allocate_tensors()

    return yolo_model, interpreter

# Panggil model
yolo_model, classifier = load_models()

# ============================================================
# 📁 Fungsi Baca Dataset
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

st.title("🧠 Dashboard Analisis Data Gambar")

# ============================================================
# 🏞️ Visualisasi Dataset
# ============================================================
if menu == "🏞️ Visualisasi Dataset":
    st.header("📊 Eksplorasi Dataset")

    dataset_type = st.selectbox("Pilih Jenis Dataset:", ["Klasifikasi Gambar", "Deteksi Objek"])
    dataset_path = "data/Klasifikasi Gambar" if dataset_type == "Klasifikasi Gambar" else "data/Object Detection"

    if not os.path.exists(dataset_path):
        st.error(f"❌ Folder '{dataset_path}' tidak ditemukan.")
    else:
        df_info = get_dataset_info(dataset_path)
        st.dataframe(df_info)

        # Grafik interaktif (Plotly)
        import plotly.express as px
        fig1 = px.bar(
        df_info,
        x="Kelas",
        y="Jumlah Gambar",
        color="Kelas",
        title="📸 Jumlah Gambar per Kelas",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig1.update_layout(showlegend=False)  # ✅ cara yang benar untuk hilangkan legend
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(
        df_info,
        x="Kelas",
        y="Rata-rata Ukuran (KB)",
        color="Kelas",
        title="💾 Rata-rata Ukuran File per Kelas",
        color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig2.update_layout(showlegend=False)  # ✅ hilangkan legend juga di sini
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
# 📷 Klasifikasi Gambar
# ============================================================
elif menu == "📷 Klasifikasi Gambar":
    st.header("📷 Klasifikasi Gambar")

    uploaded_file = st.file_uploader("Unggah gambar untuk klasifikasi:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi
        with st.spinner("🔮 Menganalisis gambar..."):
            # Dapatkan indeks input dan output
            input_details = classifier.get_input_details()
            output_details = classifier.get_output_details()

            # Set input tensor
            classifier.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

            # Jalankan inferensi
            classifier.invoke()

            # Ambil hasil output
            prediction = classifier.get_tensor(output_details[0]['index'])

            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        data_path = "data/Klasifikasi Gambar"
        class_labels = sorted(os.listdir(data_path))
        predicted_label = class_labels[class_index] if class_index < len(class_labels) else f"Class {class_index}"

        col1, col2 = st.columns(2)
        col1.markdown(f"### 🧩 Prediksi: `{predicted_label}`")
        col2.markdown(f"### 🔢 Probabilitas: `{confidence*100:.2f}%`")
        st.progress(float(confidence))

        if confidence > 0.8:
            st.success("✅ Model sangat yakin dengan hasil ini!")
        elif confidence > 0.5:
            st.warning("⚠️ Model cukup yakin, tapi perlu dicek.")
        else:
            st.error("❌ Model kurang yakin, kemungkinan salah prediksi.")

        # Visualisasi probabilitas
        fig = px.bar(
            x=class_labels, y=prediction[0],
            title="📈 Probabilitas Tiap Kelas",
            color=class_labels, color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🎯 Deteksi Objek YOLO
# ============================================================
elif menu == "🎯 Deteksi Objek (YOLO)":
    st.header("🎯 Deteksi Objek Menggunakan YOLOv8")

    uploaded_file = st.file_uploader("Unggah gambar untuk deteksi objek:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        with st.spinner("🚀 Mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()

        st.image(result_img, caption="📍 Hasil Deteksi", use_container_width=True)

        # Tabel hasil deteksi
        boxes = results[0].boxes
        if boxes:
            det_data = {
                "Label": [yolo_model.names[int(cls)] for cls in boxes.cls],
                "Confidence": [float(conf) for conf in boxes.conf],
                "Koordinat (x1,y1,x2,y2)": [box.xyxy.tolist()[0] for box in boxes]
            }
            df_det = pd.DataFrame(det_data)
            st.dataframe(df_det)
        st.success("✅ Deteksi selesai!")

# ============================================================
# ℹ️ Tentang Aplikasi — Versi Card Modern
# ============================================================
elif menu == "ℹ️ Tentang Aplikasi":
    st.header("ℹ️ Tentang Aplikasi")

    st.markdown("""
<div style='
    background: linear-gradient(135deg, #e0f2fe, #ffffff);
    border-radius: 16px;
    padding: 30px 40px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    font-family: "Segoe UI", sans-serif;
    color: #0f172a;
'>
    <h2 style='color:#1e3a8a; margin-bottom:10px;'>🎓 Dashboard UTS Praktikum Pemrograman Big Data</h2>
    <p style='font-size:16px; line-height:1.6;'>
        Dashboard ini dikembangkan sebagai bagian dari <b>Ujian Tengah Semester (UTS)</b> mata kuliah 
        <b>Praktikum Pemrograman Big Data</b>.
    </p>

    <p style='font-size:16px; line-height:1.6;'>
        Aplikasi ini bertujuan untuk mendemonstrasikan integrasi dua model analisis data gambar, yaitu:
    </p>

    <ul style='font-size:16px; line-height:1.8; margin-left:20px;'>
        <li>📷 <b>Klasifikasi Gambar</b> menggunakan model <b>Convolutional Neural Network (CNN)</b></li>
        <li>🎯 <b>Deteksi Objek</b> menggunakan model <b>YOLOv8</b></li>
    </ul>

    <p style='font-size:16px; line-height:1.6;'>
        Dashboard ini dirancang agar bersifat <b>interaktif</b> dan <b>informatif</b>, sehingga dapat digunakan 
        untuk <b>eksplorasi dan analisis data gambar (image data)</b> secara visual. Melalui implementasi ini, 
        konsep <b>Data Gambar</b> diintegrasikan dengan pendekatan <b>Big Data</b> untuk memahami bagaimana data 
        dapat diolah, dianalisis, dan divisualisasikan menggunakan teknologi modern.
    </p>

    <hr style='margin:25px 0; border:none; border-top:1px solid #93c5fd;'>

    <p style='font-size:16px; line-height:1.8;'>
        👩‍💻 <b>Dikembangkan oleh:</b> <i>Nayma Alaydia</i><br>
        📘 <b>Mata Kuliah:</b> Praktikum Pemrograman Big Data<br>
        🏫 <b>Tujuan:</b> Implementasi konsep analisis data gambar dalam konteks Big Data.
    </p>
</div>
""", unsafe_allow_html=True)

