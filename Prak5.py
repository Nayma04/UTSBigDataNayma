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

# ============================================================
# ⚙️ Konfigurasi Halaman
# ============================================================
st.set_page_config(
    page_title="Dashboard Nayma Alaydia",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 🎨 Custom CSS Style
# ============================================================
st.markdown("""
<style>
/* Font dan warna dasar */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 100%);
    color: #1e293b;
}

/* Header Utama */
h1, h2, h3 {
    font-weight: 600;
    color: #0f172a;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
}
[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 500;
}

/* Kartu konten */
.block-container {
    padding-top: 2rem;
}
div.stCard {
    background-color: #ffffff;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Tombol */
div.stButton > button {
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #2563eb, #0891b2);
    transform: scale(1.02);
}

/* Judul halaman */
.title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 📂 Fungsi Load Model
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
    ["🏞️ Visualisasi Dataset", "📷 Klasifikasi Gambar", "🎯 Deteksi Objek (YOLO)", "ℹ️ Tentang Aplikasi"]
)

# ============================================================
# 🧠 Judul Dashboard
# ============================================================
st.markdown("<div class='title'>🧠 Dashboard Analisis Data Gambar</div>", unsafe_allow_html=True)

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
        classes, counts, sizes = [], [], []
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                imgs = [f for f in os.listdir(class_path) if f.endswith((".jpg", ".png", ".jpeg"))]
                classes.append(class_name)
                counts.append(len(imgs))
                sizes.append(np.mean([os.path.getsize(os.path.join(class_path, f))/1024 for f in imgs]) if imgs else 0)
        df = pd.DataFrame({"Kelas": classes, "Jumlah Gambar": counts, "Rata-rata Ukuran (KB)": sizes})

        st.dataframe(df, use_container_width=True)

        fig = px.bar(df, x="Kelas", y="Jumlah Gambar", color="Kelas",
                     title="📸 Jumlah Gambar per Kelas",
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🖼️ Contoh Gambar")
        cols = st.columns(min(5, len(df)))
        for i, cls in enumerate(df["Kelas"][:5]):
            path = os.path.join(dataset_path, cls)
            files = [f for f in os.listdir(path) if f.endswith((".jpg", ".png", ".jpeg"))]
            if files:
                img = Image.open(os.path.join(path, files[0]))
                cols[i].image(img, caption=cls, use_container_width=True)

# ============================================================
# 📷 Klasifikasi Gambar
# ============================================================
elif menu == "📷 Klasifikasi Gambar":
    st.header("📷 Klasifikasi Gambar Menggunakan CNN")

    uploaded_file = st.file_uploader("Unggah gambar:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        img_resized = img.resize((224, 224))
        img_array = np.expand_dims(image.img_to_array(img_resized) / 255.0, axis=0)

        input_details = classifier.get_input_details()
        output_details = classifier.get_output_details()
        classifier.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
        classifier.invoke()
        pred = classifier.get_tensor(output_details[0]['index'])

        data_path = "data/Klasifikasi Gambar"
        labels = sorted(os.listdir(data_path))
        idx = np.argmax(pred)
        conf = np.max(pred)
        pred_label = labels[idx] if idx < len(labels) else "Unknown"

        st.success(f"🧩 **Prediksi:** {pred_label}  |  🔢 **Probabilitas:** {conf*100:.2f}%")
        st.progress(float(conf))

        fig = px.bar(x=labels, y=pred[0], title="📈 Distribusi Probabilitas Kelas",
                     color=labels, color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🎯 Deteksi Objek (YOLO)
# ============================================================
elif menu == "🎯 Deteksi Objek (YOLO)":
    st.header("🎯 Deteksi Objek Menggunakan YOLOv8")
    uploaded_file = st.file_uploader("Unggah gambar:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("🚀 Mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()

        st.image(result_img, caption="📍 Hasil Deteksi", use_container_width=True)
        boxes = results[0].boxes

        if boxes:
            data = {
                "Label": [yolo_model.names[int(cls)] for cls in boxes.cls],
                "Confidence": [float(conf) for conf in boxes.conf],
                "Koordinat": [box.xyxy.tolist()[0] for box in boxes]
            }
            st.dataframe(pd.DataFrame(data))
        else:
            st.warning("Tidak ada objek terdeteksi.")

# ============================================================
# ℹ️ Tentang Aplikasi
# ============================================================
elif menu == "ℹ️ Tentang Aplikasi":
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    ### 🎓 Dashboard UTS Praktikum Pemrograman Big Data  
    Aplikasi ini mendemonstrasikan integrasi dua model analisis gambar:  
    - 📷 *Klasifikasi Gambar* dengan CNN  
    - 🎯 *Deteksi Objek* dengan YOLOv8  
    
    Dashboard ini bersifat interaktif dan informatif untuk eksplorasi **data gambar** dalam konteks **Big Data**.  
    
    👩‍💻 *Dikembangkan oleh:* **Nayma Alaydia**  
    📘 *Mata Kuliah:* Praktikum Pemrograman Big Data  
    🏫 *Tujuan:* Implementasi konsep analisis data gambar dengan teknologi modern.
    """)
