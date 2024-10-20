class Config:
    def __init__(self):
        self.dim = 768 
        self.images_folder = 'images'
        self.categories_file = 'categories.txt'
        self.device = 'cpu' # gpu if you have one or more gpus
        self.model_type = 'large'
        self.model_path = 'models/magic_lens_clip_large.pkl'
        self.images_per_category = 300
        self.database = './milvus.db'
        self.collection_name = 'image_collection'