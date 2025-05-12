import streamlit as st
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load metadata
metadata = pd.read_csv("clothing_metadata.csv")

# Load model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Feature extractor
def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        return features[0]
    except:
        return None

# Load image dataset and extract features
@st.cache(allow_output_mutation=True)
def load_data():
    base_dir = "data"
    image_paths = []
    feature_list = []
    filenames = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            try:
                Image.open(img_path).verify()
                image_paths.append(img_path)
                feature = extract_features(img_path)
                if feature is not None:
                    feature_list.append(feature)
                    filenames.append(file)
            except:
                pass
    return image_paths, np.array(feature_list), filenames

# Metadata lookup
def get_metadata(filename):
    row = metadata[metadata['filename'] == filename]
    if row.empty:
        return "Unknown", "Unknown", "Unknown", "Unknown"
    row = row.iloc[0]
    return row['category'], row['size'], row['color'], row['material']

# Streamlit UI
st.title("👕 Clothing Classifier & Recommendation System")

uploaded_file = st.file_uploader("Upload a clothing image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save temp file
    temp_path = "temp.jpg"
    img.save(temp_path)

    # Metadata
    st.subheader("🧾 Clothing Information")
    category, size, color, material = get_metadata(uploaded_file.name)
    st.markdown(f"**🧥 Category**: {category}")
    st.markdown(f"**📐 Size**: {size}")
    st.markdown(f"**🎨 Color**: {color}")
    st.markdown(f"**🧵 Material**: {material}")

    # Similarity
    st.subheader("🧭 Similar Recommendations")
    image_paths, features_array, filenames = load_data()
    query_feat = extract_features(temp_path)
    if query_feat is not None:
        sims = cosine_similarity([query_feat], features_array)[0]
        top_indices = sims.argsort()[-5:][::-1]

        for idx in top_indices:
            sim_file = filenames[idx]
            sim_img = Image.open(image_paths[idx])
            st.image(sim_img, width=150, caption=sim_file)
            sim_cat, sim_size, sim_color, sim_mat = get_metadata(sim_file)
            st.markdown(f"**{sim_cat}**, {sim_size}, {sim_color}, {sim_mat}")
