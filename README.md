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
Then, the pretrained model must be downloaded by running
```
# You have to install gsutil. It is quite straightforward.
gsutil cp -R gs://gresearch/magiclens/models ./
```
Everything you need is ready nowwwwwww. Let's run app.py 
```
streamlit run streamlit_app.py
```
## Example 
Example 1: 

![Screenshot from 2024-10-21 14-10-28](https://github.com/user-attachments/assets/61907a83-32ae-4d0e-b3c3-07df4b33c27e)

Example 2: 

![Screenshot from 2024-10-21 14-25-26](https://github.com/user-attachments/assets/7d07dd7b-8e5b-477d-a728-0f5592ec2c48)


Example 3: 

![Screenshot from 2024-10-21 14-12-51](https://github.com/user-attachments/assets/bb2723ea-ac71-4512-b54f-9e3686c722dd)
