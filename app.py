import streamlit as st
import torch
import joblib
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from scipy.stats import skew
from skimage.feature import local_binary_pattern
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Penyakit Daun Singkong", layout="centered")

# Load class dan model
class_names = joblib.load("class_names.pkl")

model_pretrain = models.googlenet(pretrained=True)
model_pretrain.fc = torch.nn.Linear(1024, len(class_names))
model_pretrain.load_state_dict(torch.load("googlenet_pretrain_model.pth", map_location=torch.device('cpu')))
model_pretrain.eval()

model_nopre = models.googlenet(pretrained=False, aux_logits=False)
model_nopre.fc = torch.nn.Linear(1024, len(class_names))
model_nopre.load_state_dict(torch.load("googlenet_model.pth", map_location=torch.device('cpu')))
model_nopre.eval()

svm_model = joblib.load("svm_model.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("scaler.pkl")

# Informasi penyakit
disease_info = {
    "bacterial blight": {
        "penyebab": "Disebabkan oleh bakteri *Xanthomonas campestris pv. manihotis*.",
        "solusi": "Gunakan varietas tahan, rotasi tanaman."
    },
    "brown spot": {
        "penyebab": "Jamur *Cercosporidium henningsii*, muncul saat musim hujan.",
        "solusi": "Gunakan fungisida dan varietas tahan."
    },
    "green mite": {
        "penyebab": "Tungau merah, umum saat kemarau.",
        "solusi": "Gunakan musuh alami dan air semprot."
    },
    "mosaic": {
        "penyebab": "Virus mosaik dari kutu kebul.",
        "solusi": "Gunakan tanaman sehat, kendalikan vektor."
    },
    "healthy": {
        "penyebab": "Tanaman dalam kondisi sehat.",
        "solusi": "Lakukan monitoring dan budidaya baik."
    }
}

# Preprocessing
def preprocess_image(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(img).unsqueeze(0)

def predict_googlenet_pretrain(img):
    tensor = preprocess_image(img)
    output = model_pretrain(tensor)
    _, pred = torch.max(output, 1)
    return class_names[pred.item()]

def predict_googlenet_nopre(img):
    tensor = preprocess_image(img)
    output = model_nopre(tensor)
    _, pred = torch.max(output, 1)
    return class_names[pred.item()]

def extract_features_svm(img):
    img = cv2.resize(img, (224, 224))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8]*3, [0, 256]*3)
    hist = cv2.normalize(hist, hist).flatten()

    moments = []
    for i in range(3):
        ch = img[:, :, i]
        moments.extend([np.mean(ch), np.std(ch), skew(ch.flatten())])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)

    return np.concatenate([hist, moments, lbp_hist]).reshape(1, -1)

def predict_svm(img):
    features = extract_features_svm(img)
    scaled = scaler.transform(features)
    reduced = pca.transform(scaled)
    pred = svm_model.predict(reduced)
    return class_names[pred[0]]

# UI
st.markdown("""
    <div style='text-align: justify'>
        <h1 style='text-align: center;'>🌿 Deteksi Penyakit Daun Singkong</h1>
        <p>Prediksi otomatis menggunakan SVM dan GoogLeNet (Pretrained & Non-Pretrained)</p>
    </div>
""", unsafe_allow_html=True)

# Input
image = None
input_opt = st.radio("📤 Pilih metode input:", ["Unggah Gambar", "Gunakan Kamera"])
if input_opt == "Unggah Gambar":
    file = st.file_uploader("Unggah gambar daun", type=["jpg", "png"])
    if file: image = Image.open(file).convert("RGB")
elif input_opt == "Gunakan Kamera":
    capture = st.camera_input("Ambil gambar")
    if capture: image = Image.open(capture).convert("RGB")

if image:
    st.image(image, caption="Gambar Input", use_column_width=True)
    pred_svm = predict_svm(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    pred_nopre = predict_googlenet_nopre(image)
    pred_pre = predict_googlenet_pretrain(image)

    st.markdown("### 🧠 Hasil Deteksi Model")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📘 SVM**")
        st.info(f"Hasil: **{pred_svm.upper()}**")
        st.markdown(f"**Penyebab:** {disease_info[pred_svm]['penyebab']}")
        st.markdown(f"**Solusi:** {disease_info[pred_svm]['solusi']}")

    with col2:
        st.markdown("**🧪 GoogLeNet Tanpa Pretrain**")
        st.warning(f"Hasil: **{pred_nopre.upper()}**")
        st.markdown(f"**Penyebab:** {disease_info[pred_nopre]['penyebab']}")
        st.markdown(f"**Solusi:** {disease_info[pred_nopre]['solusi']}")

    with col3:
        st.markdown("**🤖 GoogLeNet Pretrained**")
        st.success(f"Hasil: **{pred_pre.upper()}**")
        st.markdown(f"**Penyebab:** {disease_info[pred_pre]['penyebab']}")
        st.markdown(f"**Solusi:** {disease_info[pred_pre]['solusi']}")

# Evaluasi dari file
@st.cache_resource
def load_evaluation():
    return joblib.load("eval_results.pkl")

acc_pre, acc_nopre, acc_svm, cm_pre, cm_nopre, cm_svm = load_evaluation()

st.markdown("---")
st.markdown("### 📊 **Ringkasan Akurasi & Confusion Matrix**")

col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    st.markdown("### 📘 **SVM**")
    st.markdown(f"**Akurasi:** `{acc_svm:.2%}`")
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(cm_svm, annot=True, xticklabels=class_names, yticklabels=class_names, fmt="d", cmap="Blues")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix SVM", fontsize=10)
    st.pyplot(fig)

with col2:
    st.markdown("### 🧪 **GoogLeNet Tanpa Pretrain**")
    st.markdown(f"**Akurasi:** `{acc_nopre:.2%}`")
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(cm_nopre, annot=True, xticklabels=class_names, yticklabels=class_names, fmt="d", cmap="Oranges")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix GoogLeNet (No Pretrain)", fontsize=10)
    st.pyplot(fig)

with col3:
    st.markdown("### 🤖 **GoogLeNet Pretrained**")
    st.markdown(f"**Akurasi:** `{acc_pre:.2%}`")
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(cm_pre, annot=True, xticklabels=class_names, yticklabels=class_names, fmt="d", cmap="Greens")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix GoogLeNet (Pretrained)", fontsize=10)
    st.pyplot(fig)
