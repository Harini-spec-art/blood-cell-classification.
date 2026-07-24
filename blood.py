# blood_cell_classification_app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image, ImageDataGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
import random

# ---------------------------
# 1️⃣ App Title
# ---------------------------
st.title("🩸 Blood Cell Classification - Deep Learning")
st.write("Train a CNN model on your dataset and classify blood cells.")

# ---------------------------
# 2️⃣ Dataset Paths
# ---------------------------
# Use your dataset folder
dataset_source = r"C:\Users\KEERTHI T\Desktop\New folder (2)"
dataset_dir = "dataset"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")
model_path = "blood_cell_cnn_model.h5"

# Classes
classes = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
emojis = {'Eosinophil':'🟠','Lymphocyte':'🔵','Monocyte':'🟢','Neutrophil':'⚪'}

# ---------------------------
# 3️⃣ Function: Prepare Dataset
# ---------------------------
def prepare_dataset(split_ratio=0.8):
    if not os.path.exists(dataset_source):
        st.error(f"Dataset folder not found: {dataset_source}")
        return False

    # Remove previous train/test folders
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    # Create train/test folders
    for subdir in ["train", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(dataset_dir, subdir, cls))

    # Split images by class based on filename containing class name
    for cls in classes:
        cls_images = [f for f in os.listdir(dataset_source) if cls.lower() in f.lower()]
        random.shuffle(cls_images)
        split_index = int(len(cls_images) * split_ratio)
        train_imgs = cls_images[:split_index]
        test_imgs = cls_images[split_index:]
        for img in train_imgs:
            shutil.copy(os.path.join(dataset_source,img), os.path.join(train_dir, cls,img))
        for img in test_imgs:
            shutil.copy(os.path.join(dataset_source,img), os.path.join(test_dir, cls,img))
    return True

# ---------------------------
# 4️⃣ Train CNN Model
# ---------------------------
def train_and_save_model():
    st.info("🚀 Preparing dataset and training CNN model...")
    if not prepare_dataset():
        return None

    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(128,128), batch_size=32, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(128,128), batch_size=32, class_mode='categorical')

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=10, validation_data=test_generator)
    model.save(model_path)
    st.success("✅ Model trained and saved as 'blood_cell_cnn_model.h5'")
    return model

# ---------------------------
# 5️⃣ Load Existing Model
# ---------------------------
@st.cache_resource
def load_cnn_model():
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

# ---------------------------
# 6️⃣ Train Button
# ---------------------------
if st.button("🧬 Train Model Now"):
    model = train_and_save_model()
else:
    model = load_cnn_model()
    if model is None:
        st.warning("⚠️ Model not found. Click 'Train Model Now' to create it.")

# ---------------------------
# 7️⃣ Session State for Prediction History
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Image Name","Predicted Class","Confidence (%)"])

# ---------------------------
# 8️⃣ Upload & Predict Images
# ---------------------------
st.subheader("📤 Upload Blood Cell Image(s)")
uploaded_files = st.file_uploader("Choose image(s):", type=["jpg","jpeg","png"], accept_multiple_files=True)

if model and uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        if st.button(f"🔍 Predict {uploaded_file.name}"):
            img_resized = img.resize((128,128))
            img_array = image.img_to_array(img_resized)/255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            pred_class = classes[np.argmax(prediction)]
            confidence = np.max(prediction)*100

            st.success(f"Predicted Class: **{pred_class} {emojis[pred_class]}**")
            st.info(f"Confidence: {confidence:.2f}%")

            fig, ax = plt.subplots()
            ax.pie(prediction[0],
                   labels=[f"{cls} {emojis[cls]}" for cls in classes],
                   autopct="%1.1f%%", startangle=90,
                   colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
            ax.axis('equal')
            st.pyplot(fig)

            new_row = pd.DataFrame([[uploaded_file.name, pred_class, round(confidence,2)]],
                                   columns=["Image Name","Predicted Class","Confidence (%)"])
            st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

# ---------------------------
# 9️⃣ Search Previous Predictions
# ---------------------------
st.subheader("🔎 Search Previous Predictions")
search_name = st.text_input("Enter image name:")
if search_name:
    results = st.session_state.history[st.session_state.history["Image Name"].str.contains(search_name, case=False)]
    if not results.empty:
        st.dataframe(results)
    else:
        st.warning("No results found.")

# ---------------------------
# 🔟 Full Prediction History
# ---------------------------
st.subheader("📜 Full Prediction History")
if not st.session_state.history.empty:
    st.dataframe(st.session_state.history)
    csv = st.session_state.history.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download History as CSV", data=csv, file_name="blood_cell_predictions.csv", mime="text/csv")
else:
    st.info("No predictions yet. Upload and classify images to start building history.")

st.markdown("""
---
👩‍🔬 **College Project:** Blood Cell Classification using Deep CNN  
🧠 TensorFlow + Keras | UI: Streamlit
""")
C:\Users\KEERTHI T\Desktop\New folder (2)