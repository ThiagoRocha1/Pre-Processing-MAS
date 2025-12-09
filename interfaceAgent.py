from cropAgent import Agente_Crop
from normalizeAgent import Agente_Normalizador
from shuffleAgent import Agente_Divisor
from grayAgent import Agente_Gray
from resizeAgent import Agente_Resize

class Agente_Interface:
    def __init__(self):
        print("Iniciando pre processamento das imagens")
    def processamento(self, path, crop = True, cinza=False, size=(256,256,3), split=0.75, embaralhar = True):
        if crop:
            agente_crop = Agente_Crop(size=size)
            dado_crop = agente_crop.crop_imagem(path)
        else:
            agente_res = Agente_Resize(size[0],size[1])
            dado_crop = agente_res.resize(path)
            
        if cinza:
            agente_cinza = Agente_Gray()
        agente_norm = Agente_Normalizador()
        agente_split = Agente_Divisor(split=split)
        
        
        if cinza:
            dado_cinza = agente_cinza.grayscale(dado_crop)
            dado_norm = agente_norm.normalizar_imagem(dado_cinza)
        else:
            dado_norm = agente_norm.normalizar_imagem(dado_crop)
        treino, teste = agente_split.dividir_dado(dado_norm, embaralhar)
        return treino, teste