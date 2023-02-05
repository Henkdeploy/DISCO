from src.vectorize_images import ImageVectors
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

import typer


app = typer.Typer()

@app.command()
def vector_images(folder_path:str)-> None:
    print(f"Converting images in folder : {folder_path} to vectors")
    #initialize resnet50 model
    model = tf.keras.applications.resnet50.ResNet50()
    iv = ImageVectors(model)
    #Create the vectore files
    s = iv.initialize_vectors(folder_path)

@app.command()
def compare_image(query_file_path:str,image_db_path :str, n : int = 5)-> None:
    print(f"Check  {query_file_path}")
    #initialize resnet50 model
    model = tf.keras.applications.resnet50.ResNet50()
    iv = ImageVectors(model)
    #create cevtor file for query image
    query_image = iv.initialize_vectors(query_file_path)
    #Get an array of the comaprison vectore files
    comp_arr = iv.get_comp_vectors(image_db_path)
    #Get the cosine per vectore
    cosine_arr = iv.cosine_similarity(query_image,comp_arr)
    #adding the query image at the top
    cosine_arr.append([query_file_path,100])
    #sort to get the top similarities
    cosine_arr.sort(reverse = True,key = lambda i: i[1])
    iv.most_similar(cosine_arr,n)

if __name__ == "__main__":
    app()



