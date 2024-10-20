import os
import requests
from datasets import load_dataset
from cfg import Config

def download_images():
    
    config = Config()
    images_folder = config.images_folder
    images_per_category = config.images_per_category
    categories_file = config.categories_file

    if not os.path.exists(images_folder):
        os.mkdir(images_folder)

    with open(categories_file, 'r') as f:
        for line in f:
            category = line.strip()
            
            dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}", split="full", streaming=True, trust_remote_code=True)

            images_downloaded = 0
            
            for sample in dataset:
                if images_downloaded >= images_per_category:
                    break
                
                if 'images' in sample and 'large' in sample['images'] and len(sample['images']['large']) > 0:
                    link = sample['images']['large'][0]
                    image_name = f"{category}_{images_downloaded}.jpg"
                    image_path = os.path.join(images_folder, image_name)
                    
                    try:
                        response = requests.get(link, stream=True)
                        if response.status_code == 200:
                            with open(image_path, 'wb') as img_file:
                                for chunk in response.iter_content(1024):
                                    img_file.write(chunk)
                            images_downloaded += 1
                            print(f"Downloaded {image_name}")
                        else:
                            print(f"Failed to download {image_name}")
                    except Exception as e:
                        print(f"Error downloading {image_name}: {e}")

            print(f"Finished downloading {images_downloaded} images for category: {category}")
            
            
download_images()
