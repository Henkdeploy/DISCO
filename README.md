# DISCO
Test for disco
run vectorizing
python main.py vector-images images/simple_image_retrieval_dataset/test-cases/

run compare
python main.py compare-image <PATH TO QUERY IMAGE> <PATH TO COMPARISON DB FOLDER> <--n NUMBER OF OUTPUTS (DEFUALT 5)>
example : python main.py compare-image images/simple_image_retrieval_dataset/test.png images/simple_image_retrieval_dataset/image-db/ --n 10