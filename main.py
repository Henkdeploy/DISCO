from src.vectorize_images import ImageVectors


import typer


app = typer.Typer()

@app.command()
def vector_images(folder_path:str)-> None:
    print(f"Converting images in foled: {folder_path} to vectors")
    iv = ImageVectors()
    s = iv.initialize_vectors(folder_path)
    print(s)
    
@app.command()
def get_cosine(name:str)-> None:
    print(f"Check get_cosine {name}")
    iv = ImageVectors()
    s = iv.initialize_vectors(folder_path)
    

if __name__ == "__main__":
    app()



