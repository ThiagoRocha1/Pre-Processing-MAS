import cv2
import numpy as np

class AgentMedianFilter:
    def __init__(self):
        print("Iniciando o Agente de Filtragrem")
    def filtragem(self, dado):
        lista = []
        for img in dado: 
            img_filtrada = cv2.medianBlur(img, ksize=5)
            lista.append(img_filtrada)
        dado_filtrado = np.array(lista)
        print("Filtro aplicado com sucesso")
        return dado_filtrado