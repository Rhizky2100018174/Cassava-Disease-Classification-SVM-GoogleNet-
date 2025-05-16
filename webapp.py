import streamlit as st
import torch
import joblib
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from skimage.feature import local_binary_pattern

# Set page layout
st.set_page_config(page_title="Prediksi Penyakit & Evaluasi", layout="centered")

# Path data
train_path = "D:/KULIAH IT/Skripsi/uji/train"
test_path = "D:/KULIAH IT/Skripsi/uji/test"
#class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
class_names = joblib.load("D:/KULIAH IT/Skripsi/CODE/class_names.pkl")


# Dictionary informasi penyakit

disease_info = {
    "bacterial blight": {
        "penyebab": "Disebabkan oleh bakteri *Xanthomonas campestris pv. manihotis* yang menyebar melalui air hujan dan stek terinfeksi.",
        "solusi": "Gunakan varietas tahan, rotasi tanaman, dan hindari penyebaran stek terinfeksi."
    },
    "brown spot": {
        "penyebab": "Disebabkan oleh jamur *Cercosporidium henningsii*, umum saat musim hujan.",
        "solusi": "Gunakan varietas tahan dan semprot fungisida."
    },
    "green mite": {
        "penyebab": "Disebabkan oleh tungau merah yang umum saat kemarau.",
        "solusi": "Gunakan varietas tahan dan semprotkan air atau manfaatkan musuh alami."
    },
    "mosaic": {
        "penyebab": "Disebabkan oleh virus mosaik yang ditularkan oleh kutu kebul.",
        "solusi": "Gunakan tanaman sehat dan kendalikan vektor dengan insektisida."
    },
    "healthy": {
        "penyebab": "Tanaman dalam kondisi sehat tanpa gejala penyakit.",
        "solusi": "Lakukan monitoring rutin dan budidaya yang baik."
    }
}

# Load GoogLeNet
model = models.googlenet(pretrained=True)
model.fc = torch.nn.Linear(1024, len(class_names))
model.load_state_dict(torch.load("D:/KULIAH IT/Skripsi/CODE/googlenet_model.pth", map_location=torch.device('cpu')))
model.eval()

# Load SVM dan komponen preprocessing
svm_model = joblib.load("D:/KULIAH IT/Skripsi/CODE/svm_model.pkl")
pca = joblib.load("D:/KULIAH IT/Skripsi/CODE/pca.pkl")
scaler = joblib.load("D:/KULIAH IT/Skripsi/CODE/scaler.pkl")

def preprocess_image(img, size=224):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0)

def predict_googlenet(img):
    tensor = preprocess_image(img)
    output = model(tensor)
    _, pred = torch.max(output, 1)
    return class_names[pred.item()]

def extract_features_svm(img):
    img = cv2.resize(img, (224, 224))

    # Histogram warna
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Color moments
    color_moments = []
    for i in range(3):  # B, G, R
        channel = img[:, :, i]
        color_moments.append(np.mean(channel))
        color_moments.append(np.std(channel))
        color_moments.append(skew(channel.flatten()))
    color_moments = np.array(color_moments)

    # LBP
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = 10
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    combined = np.concatenate([hist, color_moments, lbp_hist])
    return combined.reshape(1, -1)

def predict_svm(img):
    features = extract_features_svm(img)  
    scaled = scaler.transform(features)
    reduced = pca.transform(scaled)
    pred = svm_model.predict(reduced)
    return class_names[pred[0]]

# Judul utama
st.title("📷 Deteksi Penyakit Daun Singkong")
st.write("Prediksi dan evaluasi menggunakan GoogLeNet & SVM.")

# Input gambar
image = None
input_option = st.radio("Pilih sumber gambar:", ["Unggah Gambar", "Gunakan Kamera"])
if input_option == "Unggah Gambar":
    file = st.file_uploader("Unggah gambar daun", type=["jpg", "png"])
    if file:
        image = Image.open(file).convert("RGB")
elif input_option == "Gunakan Kamera":
    capture = st.camera_input("Ambil gambar")
    if capture:
        image = Image.open(capture).convert("RGB")


if image:
    # Tampilkan gambar
    st.subheader("🖼️ Gambar yang Diupload / Diambil")
    st.image(image, caption="Input Gambar", width=300)

    # Kolom untuk prediksi
    col1, col2 = st.columns(2)

    # Prediksi dengan GoogLeNet
    with col1:
        pred1 = predict_googlenet(image)
        st.subheader("🎯 GoogLeNet")
        st.success(f"Hasil: {pred1.capitalize()}")
        st.markdown(f"🌿 **Penyebab**: {disease_info[pred1]['penyebab']}")
        st.markdown(f"🌾 **Solusi**: {disease_info[pred1]['solusi']}")

    # Prediksi dengan SVM
    with col2:
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        pred2 = predict_svm(image_cv2)
        st.subheader("📘 SVM")
        st.success(f"Hasil: {pred2.capitalize()}")
        st.markdown(f"🌿 **Penyebab**: {disease_info[pred2]['penyebab']}")
        st.markdown(f"🌾 **Solusi**: {disease_info[pred2]['solusi']}")

# Evaluasi model
st.markdown("---")
st.subheader("📊 Evaluasi Model GoogLeNet & SVM")
true_labels = []
preds_googlenet = []
preds_svm = []

for label in class_names:
    folder = os.path.join(test_path, label)
    if not os.path.isdir(folder):
        continue
    for fname in os.listdir(folder):
        try:
            path = os.path.join(folder, fname)
            img = Image.open(path).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            pred1 = predict_googlenet(img)
            pred2 = predict_svm(img_cv)

            true_labels.append(label)
            preds_googlenet.append(pred1)
            preds_svm.append(pred2)
        except:
            continue

if true_labels:
    acc1 = accuracy_score(true_labels, preds_googlenet)
    acc2 = accuracy_score(true_labels, preds_svm)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"**Akurasi GoogLeNet:** {acc1:.2%}")
        cm1 = confusion_matrix(true_labels, preds_googlenet, labels=class_names)
        fig1 = plt.figure(figsize=(6, 4))
        sns.heatmap(cm1, annot=True, xticklabels=class_names, yticklabels=class_names, fmt="d", cmap="Greens")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig1)

    with col4:
        st.markdown(f"**Akurasi SVM:** {acc2:.2%}")
        cm2 = confusion_matrix(true_labels, preds_svm, labels=class_names)
        fig2 = plt.figure(figsize=(6, 4))
        sns.heatmap(cm2, annot=True, xticklabels=class_names, yticklabels=class_names, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig2)
else:
    st.info("📂 Tidak ditemukan data uji valid untuk evaluasi. Pastikan folder 'test' memiliki subfolder per kelas berisi gambar valid.")


#streamlit run "D:/KULIAH IT/Skripsi/CODE/webapp.py"
