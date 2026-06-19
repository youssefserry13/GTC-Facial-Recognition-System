# Facial Recognition System 👤

## 📌 Project Overview
Traditional security systems relying on passwords, PINs, or ID cards are vulnerable to theft and misuse.  
This project implements a **Facial Recognition System** that can identify individuals with high accuracy using deep learning.

The system is trained on publicly available face datasets, extracts embeddings using **FaceNet**, and evaluates multiple machine learning and deep learning models.  
Finally, the best-performing model is deployed via a **Streamlit web app**.

---

## 🚀 Features
- Preprocessing: resizing, normalization, augmentation (flipping, rotation, scaling)
- Embedding extraction using **FaceNet512**
- Multiple models tested (NNs, CNNs, SVM, KNN) for classification
- Evaluation using accuracy, precision, recall, and False Acceptance Rate (FAR)
- Deployment via **Streamlit app** for real-time identity prediction

---

## 🗂️ Dataset
We used publicly available datasets:
- [LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)  
- [VGGFace2](https://github.com/ox-vgg/vgg_face2)

---

## ⚙️ Methodology
1. **Data Preparation**  
   - Resize, normalize, and crop faces  
   - Augmentation (flip, rotate, scale) for robustness  

2. **EDA & Feature Building**  
   - Checked class distribution across identities  
   - Visualized embeddings with **PCA** and **t-SNE**  

3. **Models Trained**
   - **FaceNet (Facenet512)** → used to generate embeddings from face images  
   - **Neural Networks (NNs)**  
     - Baseline NN (2 hidden layers + dropout)  
     - Improved NN (deeper, EarlyStopping)  
     - Improved NN v2 (512 → 256 → 128 with LR scheduler)  
     - Alternative NN (BatchNormalization + Dropout)  
   - **Convolutional Neural Networks (CNNs)**  
     - Custom CNN trained directly on processed face images  
     - Transfer Learning with **MobileNetV2** + fine-tuned dense layers  
   - **Classical ML Models (on embeddings)**  
     - Support Vector Machine (SVM: linear & RBF kernels, GridSearchCV tuned)  
     - K-Nearest Neighbors (KNN)  

4. **Evaluation**  
   - Metrics: Accuracy, Precision, Recall, False Acceptance Rate (FAR)  
   - Final best model chosen based on results (insert your best model + metrics here)  

5. **Deployment**  
   - Built a Streamlit app: upload an image → preprocess → predict → output identity  

---

## 📊 Results
> 🔽 Add your results here after running experiments:

- Baseline NN: Accuracy = 92%  
- Improved NN (v1): Accuracy = 94%  
-Improved NN (v2) = 92%
- Alternative NN = 92%
- Fine-tuned MobileNetV2= 60%
- kernal SVM= 93%
- **Best Model:**  Improved NN (v1) with Accuracy = 94%  

---

## 📁 Repository Structure
- `app.py`: Streamlit application for real-time identity prediction.
- `docs/`: Project documentation and presentations.
- `models/`: Trained model files.
- `results/`: Evaluation results and metrics.
- `phase1_data_preparation.ipynb`: Data preprocessing and augmentation.
- `phase2_eda_and_feature_building.ipynb`: Exploratory Data Analysis and feature extraction.
- `phase3_model_training_and_evaluation.ipynb`: Model training, tuning, and evaluation.

## 💻 Streamlit App
To run the app locally:
```bash
streamlit run app.py
```

