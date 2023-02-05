
from src.vectorize_images import ImageVectors
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import numpy as np
import unittest

model = tf.keras.applications.resnet50.ResNet50()

class Test(unittest.TestCase):
    def test_cosine_same_image(self):
        #Test when passing the same image vectore if cosine returns 1
        query_vector1 = np.load("test/test_png.npy")
        query_vector2 = np.load("test/test_png.npy")
        self.assertEqual(int(round(ImageVectors(model).cosine_func(query_vector1,query_vector2),0)), 1)
        
    def test_cosine_diff_image(self):
        #Test when passing different image vectors if cosine doesnt returns 1
        query_vector1 = np.load("test/test_png.npy")
        query_vector2 = np.load("test/leopard_jpg.npy")
        self.assertEqual(int(round(ImageVectors(model).cosine_func(query_vector1,query_vector2),0)), 1)
