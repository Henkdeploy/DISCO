import os
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from numpy.linalg import norm 
import cv2 
import matplotlib.pyplot as plt
import fnmatch
import math  
import time


class ImageVectors:
    model = None
    def __init__(self, arg):
        ImageVectors.model = arg
        
    def pre_process(self,file_path) -> np.ndarray:
        #Do all the preproccessing needed for geting the feature array
        image = tf.keras.utils.load_img(file_path, target_size=(224, 224))
        ppx = tf.keras.utils.img_to_array(image)
        ppx = np.expand_dims(ppx, axis=0)
        ppx = tf.keras.applications.resnet50.preprocess_input(ppx)
        return ppx
    
    def convert_to_vector(self,file_path:str,file_name):
        x = self.pre_process(file_path)
        #get the feature vectore from the preprocessed arrays
        predictions = self.model.predict(x)
        flattened_features = predictions.flatten()
        #save array in a pickle
        np.save(f"images/tmp/{file_name.replace('.','_')}", flattened_features, allow_pickle=True, fix_imports=True)
        
    def initialize_vectors(self,folder_path:str) -> str:
        res = []
        #vectorize a file 
        if os.path.isfile(folder_path):
            res.append(folder_path)
            self.convert_to_vector(f"{folder_path}",os.path.basename(folder_path))
            return folder_path
        else: #vectorize an entire folder 
            #Count how many files in folder for progress calc
            files = len(fnmatch.filter(os.listdir(folder_path), '*.*'))
            for count , path in enumerate(os.listdir(folder_path)):
                #Show progress cal so poeple know its not haging
                print(f"Percentage done : {(count/files)*100}%")
                if os.path.isfile(os.path.join(folder_path, path)):
                    #Append the file path if its a proper image filke
                    try:
                        self.convert_to_vector(f"{folder_path}{path}",path)
                        res.append(path)
                    except:
                        print(f"{os.path.join(folder_path, path)} is not valid")
            return res[0]

    
    def get_comp_vectors(self,image_db_path:str) -> np.ndarray:
        res = []
        #Count how many files in folder for progress calc
        files = len(fnmatch.filter(os.listdir(image_db_path), '*.*'))
        for count, path in enumerate(os.listdir(image_db_path)):
            #Show progress cal so poeple know its not haging
            print(f"Percentage done : {(count/files)*100}%")
            if os.path.isfile(os.path.join(image_db_path, path)):
                #Check if image has a vectorized file
                if os.path.isfile(f"images/tmp/{path.replace('.','_')}.npy"):
                    res.append([f"images/tmp/{path.replace('.','_')}.npy",os.path.join(image_db_path, path)])
                else:
                    #if not then vectorize it if possible
                    print("NO FILE TRY TO VECTORIZE")
                    try:
                        self.convert_to_vector(f"{image_db_path}{path}",path)
                        res.append([f"images/tmp/{path.replace('.','_')}.npy",os.path.join(image_db_path, path)])
                    except:
                        print(f"{os.path.join(image_db_path, path)} is not valid")       
        return res
    
    def cosine_func(self,query_vector :np.ndarray,comp_vector :np.ndarray) -> float:
        #Calculate the cosine difference
        cosine = np.dot(query_vector,comp_vector)/(norm(query_vector)*norm(comp_vector))
        return cosine
        
        
        
    def cosine_similarity(self,query_image_file : str,image_vector_files:np.ndarray)-> np.ndarray:
        query_vector = f"images/tmp/{os.path.basename(query_image_file).replace('.','_')}.npy"
        cosine_arr = []
        query_vector = np.load(query_vector)
        #calculate cosine per vector
        for comp_vec_path in image_vector_files:
            comp_vector = np.load(comp_vec_path[0])
            cosine = self.cosine_func(query_vector,comp_vector)
            cosine_arr.append([comp_vec_path[1],cosine])
        return cosine_arr
            
            
            
    def most_similar(self,cosine_arr:np.ndarray,n:int):
        for count, i in enumerate(cosine_arr):
          #  Plots a n by 6 image collage of top n picks
            plt.subplot(int(round(math.ceil(n/5),0)),6,count+1)
            img = cv2.imread(i[0])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            resize = cv2.resize(img, [224,224], interpolation = cv2.INTER_AREA)
            if int(i[1]) == 100:
                title = f"Query image \n {os.path.basename(i[0])} "
                filename = f"output/{os.path.basename(i[0]).replace('.','_')}_collage_{round(time.time(),0)}.jpeg"
            else:
                title = f"{os.path.basename(i[0])} , \nscore: {str(round(i[1],2))}"
                
            plt.title(title, fontdict=None, loc='center', wrap=True , fontsize=8) 
            plt.imshow(resize)
            plt.axis('off')
            if count == n:
                break
        plt.suptitle(f"Compare images to {os.path.basename(cosine_arr[0][0])}", fontsize=12)
        
        plt.savefig(filename)
        

        