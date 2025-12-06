import os
import tensorflow as tf
import numpy as np
from PIL import Image


class Agente_Crop:
    def __init__(self, size = (256, 256, 3)):
        self.size = size
        print("Agente de Recorte iniciado")
    def crop_imagem(self, path,folder_dir):
        lista_img = []
        for images in os.listdir(path):
            if (images.endswith(".png")):
                img = Image.open(folder_dir +'/'+ images)
                imagem_np = np.array(img)
                x = tf.image.random_crop(imagem_np, size=self.size, seed=None, name=None)
                image_np = x.numpy()
                lista_img.append(image_np)
        dado_crop = np.array(lista_img)
        print(f"Imagem cropeada com sucesso.")
        return dado_crop 