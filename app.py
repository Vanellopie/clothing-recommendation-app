import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from google import genai
from google.genai import types
from brave import Brave

# --- Load metadata ---
metadata = pd.read_csv("clothing_metadata.csv")

# --- Load model ---
@st.cache_resource
def load_model():
    base = VGG16(weights='imagenet')
    return Model(inputs=base.input, outputs=base.get_layer('fc1').output)

model = load_model()

# --- Feature extraction function ---
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

# --- Extract features for all images ---
def extract_features_for_all_images(valid_image_paths):
    features_list = []
    valid_filenames = []
    
    for img_path in valid_image_paths:
        features = extract_feature_from_upload(img_path)
        if features is not None:
            features_list.append(features)
            # valid_filenames.append(os.path.basename(img_path))
            valid_filenames.append(img_path)


    features_array = np.array(features_list)
    return features_array, valid_filenames

# --- Chatbot Configuration ---
api_key = "AIzaSyBpmVO6jwUvt44hS_Q1TwVPmVnKqBBPjFM"
brave_api_key = "BSAP1ZmJl9wMXKDvGnGM78r9__i_VuG"

SYSTEM_PROMPT = """
You are a helpful, professional, and knowledgeable assistant for [Comfy Pajama Shop]. Your role is to support customers by answering their questions based only on the information provided below. Please do not answer questions unrelated to the store.

Guidelines for Responses:
Answer in the language that the customer uses.

Be polite, friendly, and clear.

Do not refer customers to external websites or phone numbers unless they explicitly ask.

Do not mention specific individuals unless the customer brings them up first.

Do not say that you are a bot.

If you're unsure or the question is outside the provided information, say:
“Thank you for your question. I will forward it to our customer support team and get back to you.”

Predefined Questions and Answers:
🛌 General Information

What is your store's name?
→ Our store is called Comfy Pajama Shop.

Where are you located?
→ We are located at Altjin Bumbugur hudaldaanii tuviin, 2 davhart, 222 toot.

What types of clothing do you sell?
→ We offer a variety of clothes, including pajamas, maternity (pregnancy) pajamas, home wear, and sleepwear for both men and women. Sizes range from M to 4XL.

What are your store hours?
→ We’re open Monday to Sunday, 10:00 AM – 7:30 PM, and we’re closed on Thursday.

Do you offer online shopping?
→ Sorry, we do not offer online shopping at the moment.

Can I return or exchange my purchase?
→ Yes, we accept returns or exchanges within 2 days of purchase with the original receipt. Items must be unworn and in original condition.

🛍️ Discounts

Are there any ongoing sales?
→ Yes! We offer seasonal sales and special discounts throughout the year. Please visit the store for current promotions.

👕 Product Information

How can I find my size?
→ You can check our size guide on the product page, or our friendly staff will assist you in-store to find the perfect fit.

Are your clothes machine washable?
→ Yes, all of our clothes are machine washable. Please check the care label on each item for specific instructions.

💳 Payment

What payment methods do you accept?
→ We accept credit/debit cards, mobile payments, and cash in-store.

🧸 About Us

How long has your store been in business?
→ Comfy Pajama Shop has been proudly serving the community for over 3 years, providing high-quality fashion and excellent customer service.

Do you offer wholesale purchasing?
→ Yes, we offer wholesale purchasing for businesses. Please contact our sales team for more information.

🌙 Reminder:
Only answer questions based on the information above.
If a question falls outside this scope, respond with:
“Thank you for your question. I will forward it to our customer support team and get back to you.”
"""

def query_brave(query: str) -> str:
    brave = Brave(api_key=brave_api_key)
    search_results = brave.search(q=query, count=5, raw=True)
    return search_results['web']

def initialize_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)

def get_gemini_response(client: genai.Client, messages):
    last_user_message = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
    
    search_indicators = ["what is", "who is", "how to", "tell me about", "search for", "find"]
    should_search = any(indicator in last_user_message.lower() for indicator in search_indicators)
    
    contents = [msg["content"] for msg in messages]
    
    if should_search:
        try:
            web_results = query_brave(last_user_message)
            if web_results:
                search_context = "\n\nHere are some relevant web search results:\n" + str(web_results)
                contents[-1] = contents[-1] + search_context
        except Exception as e:
            st.warning(f"Web search failed: {str(e)}")
    
    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=2048,
        )
    )
    return response.text

# --- Streamlit UI with Tabs ---
st.title("Comfy Pajama Shop Assistant")

tab1, tab2, tab3 = st.tabs(["👕 Clothing Classifier", "🔁 Similar Products", "💬 Chatbot"])

# --- Tab 1: Clothing Classifier ---
with tab1:
    st.subheader("Upload a clothing image to view details")
    uploaded_file = st.file_uploader("Upload a clothing image", type=['heic', 'jpg', 'jpeg', 'png'])

    if uploaded_file:
        filename = uploaded_file.name
        info = metadata[metadata['filename'] == filename]

        col1, col_spacer, col2 = st.columns([1, 0.4, 1.5])
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown("### 📄 Clothing Info")
            if not info.empty:
                st.write("**Name:**", filename)
                st.write("**Category:**", info['category'].values[0])
                st.write("**Size:**", info['size'].values[0])
                st.write("**Color:**", info['color'].values[0])
                st.write("**Material:**", info['material'].values[0])
                st.write("**Price:**", info['price'].values[0])
            else:
                st.warning("Metadata not found for this file.")

# --- Tab 2: Similar Products ---
with tab2:
    st.subheader("Find similar clothing items")

    if uploaded_file:
        query_feat = extract_feature_from_upload(uploaded_file)
        if query_feat is not None:
            try:
                data_folder = "data"
                subfolders = ["halad", "hoslol", "hoslol_silk", "short"]
                valid_image_paths = []

                for subfolder in subfolders:
                    subfolder_path = os.path.join(data_folder, subfolder)
                    if os.path.exists(subfolder_path):
                        for img_file in os.listdir(subfolder_path):
                            if img_file.endswith(".png"):
                                img_path = os.path.join(subfolder_path, img_file)
                                valid_image_paths.append(img_path)

                features_array, valid_filenames = extract_features_for_all_images(valid_image_paths)
                sims = cosine_similarity([query_feat], features_array)[0]
                top_n = min(5, len(sims))  # Avoid index error if fewer than 5 items
                top_indices = sims.argsort()[-top_n:][::-1]

                st.markdown("### Top 3 Similar Products:")
                display_n = min(3, top_n)
                cols = st.columns(display_n)

                for i in range(display_n):
                    idx = top_indices[i]
                    with cols[i]:
                        st.image(valid_filenames[idx], width=100,
                                 caption=os.path.basename(valid_filenames[idx]))

            except Exception as e:
                st.error(f"Error finding similar products: {e}")
        else:
            st.warning("Please upload a valid image first.")
    else:
        st.info("Upload an image in the 'Clothing Classifier' tab to get started.")


# --- Tab 3: Chatbot ---
with tab3:
    st.subheader("Ask anything about Comfy Pajama Shop!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    client = initialize_gemini_client(api_key)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything about Comfy Pajama Shop!"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response = get_gemini_response(client, st.session_state.messages)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")
