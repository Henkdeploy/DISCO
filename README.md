# DISCO Image retrieval

This is my code for DISCO's image retrieval 

## Installation

Install packages using the requirements text

```bash
pip install -r requirements.txt
```

## Runnint the vectorizing

### Step 1:
Create a folder called images in the root folder of the repository
```bash
mkdir images
```

### Step 2:
Unzip the image db into the image folder

### Step 3:

To vectorize the images in the image db run the following comand 

**Note**  This may take up to 12mins for 4000 images

run:
```bash
python main.py vector-images <PATH TO IMAGE DB>
```

example
```bash
python main.py vector-images images/simple_image_retrieval_dataset/image-db/
```

### Step 4:
Choose an image to comapre to the db

run
```bash
python main.py compare-image <PATH TO QUERY IMAGE> <PATH TO COMPARISON DB FOLDER> <--n NUMBER OF OUTPUTS (DEFUALT 5)>
```
example:
```bash
python main.py compare-image images/simple_image_retrieval_dataset/test.png images/simple_image_retrieval_dataset/image-db/ --n 10
```
```bash
python main.py compare-image images/simple_image_retrieval_dataset/test.png images/simple_image_retrieval_dataset/image-db/
```
**Note** If any new images that hasnt been vecorized gets into the db this will vectorize them aswell

### Step 5:
The collage will be saved in the output folder in the root of the repo under the name of the query image and a time stamp


## To run py test run:
```bash
pytest 
```
