import streamlit as st
import torch
import joblib
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from scipy.stats import skew
from skimage.feature import local_binary_pattern

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Penyakit Singkong", layout="centered")

# Load class dan model
class_names = joblib.load("class_names.pkl")
model = models.googlenet(pretrained=True)
model.fc = torch.nn.Linear(1024, len(class_names))
model.load_state_dict(torch.load("googlenet_model.pth", map_location=torch.device('cpu')))
model.eval()

svm_model = joblib.load("svm_model.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("scaler.pkl")

# Info penyakit
disease_info = {
    "bacterial blight": {
        "penyebab": "Disebabkan oleh bakteri *Xanthomonas campestris pv. manihotis*.",
        "solusi": "Gunakan varietas tahan dan hindari stek terinfeksi."
    },
    "brown spot": {
        "penyebab": "Disebabkan oleh jamur *Cercosporidium henningsii*.",
        "solusi": "Gunakan varietas tahan dan semprot fungisida."
    },
    "green mite": {
        "penyebab": "Disebabkan oleh tungau merah.",
        "solusi": "Gunakan varietas tahan dan semprot air atau manfaatkan musuh alami."
    },
    "mosaic": {
        "penyebab": "Disebabkan oleh virus mosaik melalui kutu kebul.",
        "solusi": "Gunakan tanaman sehat dan kendalikan kutu putih."
    },
    "healthy": {
        "penyebab": "Tanaman sehat tanpa gejala penyakit.",
        "solusi": "Lakukan pemantauan dan budidaya optimal."
    }
}

# Fungsi prediksi GoogLeNet
def preprocess_image(img, size=224):
    img = np.array(img)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(img).unsqueeze(0)

def predict_googlenet(img):
    tensor = preprocess_image(img)
    output = model(tensor)
    _, pred = torch.max(output, 1)
    return class_names[pred.item()]

# Fungsi prediksi SVM
def extract_features_svm(img):
    img = cv2.resize(img, (224, 224))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
    hist = cv2.normalize(hist, hist).flatten()
    color_moments = []
    for i in range(3):
        ch = img[:, :, i]
        color_moments += [np.mean(ch), np.std(ch), skew(ch.flatten())]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return np.concatenate([hist, color_moments, lbp_hist]).reshape(1, -1)

def predict_svm(img):
    features = extract_features_svm(img)
    scaled = scaler.transform(features)
    reduced = pca.transform(scaled)
    pred = svm_model.predict(reduced)
    return class_names[pred[0]]

# Tampilan antarmuka
st.title("📷 Deteksi Penyakit Daun Singkong")
st.write("Prediksi penyakit daun singkong menggunakan GoogLeNet dan SVM.")

# Input gambar
image = None
input_option = st.radio("Pilih metode input gambar:", ["Unggah Gambar", "Gunakan Kamera"])
if input_option == "Unggah Gambar":
    file = st.file_uploader("Unggah gambar daun", type=["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file).convert("RGB")
elif input_option == "Gunakan Kamera":
    capture = st.camera_input("Ambil gambar")
    if capture:
        image = Image.open(capture).convert("RGB")

# Jika gambar tersedia
if image:
    st.subheader("🖼️ Gambar yang Diupload")
    st.image(image, caption="Input Gambar", width=300)

    col1, col2 = st.columns(2)

    with col1:
        pred1 = predict_googlenet(image)
        st.subheader("🎯 GoogLeNet")
        st.success(f"Hasil: {pred1.capitalize()}")
        st.markdown(f"🌿 **Penyebab**: {disease_info[pred1]['penyebab']}")
        st.markdown(f"🌾 **Solusi**: {disease_info[pred1]['solusi']}")

    with col2:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        pred2 = predict_svm(img_cv)
        st.subheader("📘 SVM")
        st.success(f"Hasil: {pred2.capitalize()}")
        st.markdown(f"🌿 **Penyebab**: {disease_info[pred2]['penyebab']}")
        st.markdown(f"🌾 **Solusi**: {disease_info[pred2]['solusi']}")
