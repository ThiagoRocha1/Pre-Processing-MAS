import numpy as np
import cv2

class Agente_Gray:
    def __init__(self):
        print("Agente Grayscale iniciado")
    def grayscale(self, dado):
        lista = []
        for i in range(dado.shape[0]):
            imagem_cinza = cv2.cvtColor(dado[i], cv2.COLOR_RGB2GRAY)
            lista.append(imagem_cinza)
        dado_cinza = np.array(lista)
        print("Imagem alterada para preto e branco com sucesso")
        return dado_cinza