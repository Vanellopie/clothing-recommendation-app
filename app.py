import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# --- Load metadata & features ---
metadata = pd.read_csv("clothing_metadata.csv")
features = np.load("features.npy")
paths = pd.read_csv("paths.csv")[0].tolist()

# --- Load model ---
@st.cache_resource
def load_model():
    base = VGG16(weights='imagenet')
    return Model(inputs=base.input, outputs=base.get_layer('fc1').output)

model = load_model()

def extract_feature_from_upload(img_file):
    try:
        img = Image.open(img_file).convert("RGB")
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = model.predict(x)
        return feat.flatten()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- Streamlit UI ---
st.title("👕 Clothing Classifier & Recommender")
uploaded_file = st.file_uploader("Upload a clothing image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=200)
    filename = uploaded_file.name
    info = metadata[metadata['filename'] == filename]

    if not info.empty:
        st.markdown("### 📄 Clothing Info")
        st.write("**Name:**", filename)
        st.write("**Category:**", info['category'].values[0])
        st.write("**Size:**", info['size'].values[0])
        st.write("**Color:**", info['color'].values[0])
        st.write("**Material:**", info['material'].values[0])
    else:
        st.warning("Metadata not found for this file.")

    st.markdown("### 🔁 Similar Products")
    query_feat = extract_feature_from_upload(uploaded_file)

    if query_feat is not None:
        sims = cosine_similarity([query_feat], features)[0]
        top_indices = sims.argsort()[-5:][::-1]

        cols = st.columns(5)
        for i, idx in enumerate(top_indices):
            with cols[i]:
                st.image(paths[idx], width=100, caption=os.path.basename(paths[idx]))
