import os
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

class ImageVectors:
    def convert_to_vector(self,file_path:str) -> str:
        print(f"file path {file_path}")
        image = tf.keras.utils.load_img(file_path, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.resnet50.preprocess_input(x)
        model = tf.keras.applications.resnet50.ResNet50()
        predictions = model.predict(x)
        flattened_features = predictions.flatten()
       # np.save("filename", flattened_features, allow_pickle=True, fix_imports=True)
        
        
        
    def initialize_vectors(self,folder_path:str) -> int:
        res = []
    
        # Iterate directory
        for path in os.listdir(folder_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(folder_path, path)):
                res.append(path)
                self.convert_to_vector(f"{folder_path}{path}")
        print(f" {folder_path} has {len(res)} images {res[2]}")
        return len(res)