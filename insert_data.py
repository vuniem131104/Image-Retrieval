from pymilvus import MilvusClient
import numpy as np 
import os 
from tqdm.auto import tqdm
from cfg import Config
from retrieval import ImageRetrievalModel, load_model_params

config = Config()

images_folder = config.images_folder

def get_model():
    encoder, params = load_model_params(config.model_type, config.model_path)
    model = ImageRetrievalModel(params, encoder)
    return model

model = get_model()
fixed_tokens = np.array([[49406, 49407,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0]], dtype=np.int32)


list_images = os.listdir(images_folder)
dict_images = dict()

for image_name in tqdm(list_images, desc="Generating Embeddings"):
    image_path = os.path.join(images_folder, image_name)
    output = model.retrieve(image_path, fixed_tokens)
    dict_images[image_name] = output

print(f'Generated {len(dict_images)} image embeddings')

client = MilvusClient(config.database)

client.create_collection(config.collection_name, auto_id=True, dimension=768, enable_dynamic_field=True,)

client.insert(collection_name=config.collection_name, data=[{'image_path': k, 'vector': v} for k, v in dict_images.items()])