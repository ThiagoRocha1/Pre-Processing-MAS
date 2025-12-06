import cropAgent as Agente_Crop
import normalizeAgent as Agente_Normalizador
import shuffleAgent as Agente_Divisor
import grayAgent as Agente_Gray

class Agente_Interface:
    def __init__(self):
        print("Iniciando pre processamento das imagens")
    def processamento(self, path, cinza=False, size=(256,256,3), split=0.75, embaralhar = True):
        folder_dir = 'imagens/DIV2K_train_HR/DIV2K_train_HR'

        agente_crop = Agente_Crop(size=size)
        if cinza:
            agente_cinza = Agente_Gray()
        agente_norm = Agente_Normalizador()
        agente_split = Agente_Divisor(split=split)
        
        dado_crop = agente_crop.crop_imagem(path,folder_dir)
        if cinza:
            dado_cinza = agente_cinza.grayscale(dado_crop)
            dado_norm = agente_norm.normalizar_imagem(dado_cinza)
        else:
            dado_norm = agente_norm.normalizar_imagem(dado_crop)
        treino, teste = agente_split.dividir_dado(dado_norm, embaralhar)
        return treino, teste