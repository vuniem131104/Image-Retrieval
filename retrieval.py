from model import MagicLens
import pickle
import jax 
import jax.numpy as jnp
from flax import serialization 
from PIL import Image
import numpy as np
from cfg import Config
import clip

config = Config()

device = config.device
jax.config.update("jax_platform_name", device)

def process_image(image_path, size):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    img = np.array(img) / 255.0
    img = img[np.newaxis, ...]
    return img


def load_model_params(model_type, model_path):
    encoder = MagicLens(model_type)
    rng = jax.random.PRNGKey(0)
    dummy_input = {
        "ids": jnp.ones((1, 1, 77), dtype=jnp.int32),
        "image": jnp.ones((1, 224, 224, 3), dtype=jnp.float32),
    }
    params = encoder.init(rng, dummy_input)
    print("Model initialized")

    with open(model_path, "rb") as f:
        model_bytes = pickle.load(f)
    params = serialization.from_bytes(params, model_bytes)
    return encoder, params


class ImageRetrievalModel:
    def __init__(self, params, encoder):
        self.params = params
        self.encoder = encoder

    def retrieve(self, image_path, tokens):
        img = process_image(image_path, 224)
        res = self.encoder.apply(self.params, {"ids": tokens, "image": img})
        return list(np.array(res["multimodal_embed"])[0])
    
    
# if __name__ == '__main__':
#     config = Config()
#     encoder, params = load_model_params(config.model_type, config.model_path)
#     model = ImageRetrievalModel(params, encoder)
#     print(len(model.retrieve("images.jpeg", clip.tokenize("").numpy())))
    