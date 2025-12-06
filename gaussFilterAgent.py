import numpy as np

class Agente_Filtro_Gauss:
    def __init__(self):
        print("Iniciando o Agente de Filtragrem")
    def filtragem(self, dado, kernel = 5):
        lista = []
        for img in dado: 
            img_filtrada = cv2.GaussianBlur(img,(kernel,kernel),0)
            lista.append(img_filtrada)
        dado_filtrado = np.array(lista)
        print("Filtro aplicado com sucesso")
        return dado_filtrado