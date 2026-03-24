import os
import torch
import requests
from zipfile import ZipFile
from io import BytesIO

# On définit le chemin local dans ton projet
dataset_path = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
target_folder = './data' # Dossier local au projet

print(f"Début du téléchargement vers {target_folder}...")

response = requests.get(dataset_path)
if response.status_code == 200:
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall(target_folder)
    print('Téléchargement et extraction terminés dans /data !')
else:
    print(f"Erreur lors du téléchargement : {response.status_code}") 
    