import streamlit as st
from PIL import Image
import os 
import clip
from pymilvus import MilvusClient
from cfg import Config
from retrieval import ImageRetrievalModel, load_model_params

def image_to_base64(image):
    import io
    import base64
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


@st.cache_resource
def get_model():
    config = Config()
    encoder, params = load_model_params(config.model_type, config.model_path)
    model = ImageRetrievalModel(params, encoder)
    return model

@st.cache_resource
def get_milvus_client():
    config = Config()
    client = MilvusClient(config.database)
    return client

st.set_page_config(page_title="Multimodal Image Search", layout="wide")

st.title("Multimodal Image Search")

st.sidebar.header("Enter your Query")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

text_query = st.sidebar.text_input("Enter Text Query")

col1, col2 = st.sidebar.columns([1, 1])  

with col1:
    search_clicked = st.button("Search")
    
with col2:
    rerank_clicked = st.button("Rerank")

model = get_model()
client = get_milvus_client()

if search_clicked:
    st.write("Searching...")
    if uploaded_file and text_query:
        with open("tmp.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())
            
    query_tokens = clip.tokenize(text_query).numpy()
    query_vector = model.retrieve("tmp.jpg", query_tokens)
    res = client.search(collection_name='image_collection', data=[query_vector], output_fields=['image_path'], 
                        limit=9, search_params={"metric_type": "COSINE", "params": {}})[0]
    
    retrieved_images = [hit.get("entity").get("image_path") for hit in res]
    
    num_cols = 3  
    rows = len(retrieved_images) // num_cols + int(len(retrieved_images) % num_cols != 0)

    st.write("### Retrieved Images")
    for row in range(rows):
        cols = st.columns(num_cols)
        for col_idx, img_path in enumerate(retrieved_images[row * num_cols:(row + 1) * num_cols]):
            idx = row * num_cols + col_idx + 1  
            
            if img_path and os.path.exists(os.path.join("images", img_path)):
                image = Image.open(os.path.join("images", img_path))

                desired_size = (224, 224)  
                image = image.resize(desired_size)

                with cols[col_idx]:
                    st.write(f"Top {idx}")  
                    img_html = f'<div style="border: 2px solid black; padding: 5px; display: inline-block;">' \
                               f'<img src="data:image/png;base64,{image_to_base64(image)}" width="300" height="300"/>' \
                               f'</div>'
                    st.markdown(img_html, unsafe_allow_html=True)
            else:
                cols[col_idx].write(f"Image {idx}: Image not found")

if rerank_clicked:
    st.write("Reranking...")

