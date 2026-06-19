import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import joblib
from deepface import DeepFace
import os

# ===============================
# 1. Load Model + Encoder
# ===============================
MODEL_PATH = os.path.join("models", "best_model_improved_v1.h5")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

# Note: These files are not currently in the repository, but we've updated the paths
# to point to the models/ directory for future placement.
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
else:
    st.warning("Model or Label Encoder not found in 'models/' directory.")

# ===============================
# 2. Streamlit App
# ===============================
st.title("👤 Face Recognition App")
st.write("Upload a face image and the model will predict the identity.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to numpy image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    try:
        # ✅ Generate embeddings with Facenet512
        embedding = DeepFace.represent(
            img_path=img,
            model_name="Facenet512",
            enforce_detection=False
        )[0]["embedding"]
        embedding = np.array(embedding).reshape(1, -1)  # Shape (1,512)

        # Prediction
        pred = model.predict(embedding, verbose=0)
        pred_class = np.argmax(pred, axis=1)[0]
        pred_name = label_encoder.classes_[pred_class]

        # Display results
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption=f"Prediction: {pred_name}",
                 use_column_width=True)
        st.success(f"✅ Predicted Identity: {pred_name}")

    except Exception as e:
        st.error(f"⚠️ Face embedding failed: {str(e)}")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption="Uploaded Image",
                 use_column_width=True)