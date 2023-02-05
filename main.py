from src.vectorize_images import ImageVectors
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

import typer


app = typer.Typer()

@app.command()
def vector_images(folder_path:str)-> None:
    print(f"Converting images in folder : {folder_path} to vectors")
    model = tf.keras.applications.resnet50.ResNet50()
    iv = ImageVectors(model)
    s = iv.initialize_vectors(folder_path)

@app.command()
def compare_image(query_file_path:str,image_db_path :str, n : int = 5)-> None:
    print(f"Check  {query_file_path}")
    model = tf.keras.applications.resnet50.ResNet50()
    iv = ImageVectors(model)
    query_image = iv.initialize_vectors(query_file_path)
    comp_arr = iv.get_comp_vectors(image_db_path)
    cosine_arr = iv.cosine_similarity(query_image,comp_arr)
    cosine_arr.append([query_file_path,100])
    cosine_arr.sort(reverse = True,key = lambda i: i[1])
    iv.most_similar(cosine_arr,n)

if __name__ == "__main__":
    app()



