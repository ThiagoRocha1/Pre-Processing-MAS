import os
import numpy as np
import cv2
from PIL import Image

class AgentResize:
    def __init__(self, new_width=256, new_height=256):
        self.new_width = new_width
        self.new_height = new_height
        print("Agente Resize iniciado")
    def resize(self, path):
        lista_img = []
        for images in os.listdir(path):
            if (images.endswith(".png")):
                img = Image.open(path +'/'+ images)
                imagem_np = np.array(img)
                res_image = cv2.resize(imagem_np, (self.new_width, self.new_height))
                lista_img.append(res_image)
        dado_res = np.array(lista_img)
        print(f"Imagem redimensionado com sucesso.")
        return dado_res
