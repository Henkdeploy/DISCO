import os

class ImageVectors:
    def convert_to_vector(self,file_path:str) -> str:
        print(f"file path {file_path}")
        
        
        
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