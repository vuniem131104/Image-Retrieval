# Image Retrieval With Magiclens
## Description
This project leverages the power of [Magiclens](https://github.com/google-deepmind/magiclens) to retrieve similar images from given image and text from users and build interface thanks to streamlit in python
## Setup
You need to open terminal and run 
```
pip install -r requirements.txt
```
## Run Everything
Firstly, you need to download images to your local computer. I am using a subset of https://github.com/hyp1231/AmazonReviews2023 which includes approximately 10000 images in 33 different categories, such as applicances, beauty and personal care, clothing, sports and outdoors, etc.
Downloading by running
```
python download_images.py
```
Next, you need to push all the embeddings which was processed by [Magiclens](https://github.com/google-deepmind/magiclens) into Milvus
```
python insert_data.py
```
Everything you need is ready nowwwwwww. Let's run app.py 
```
streamlit run streamlit_app.py
```
## Example 
loading....
