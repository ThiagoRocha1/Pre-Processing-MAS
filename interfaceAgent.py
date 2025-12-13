from cropAgent import AgentCrop
from normalizeAgent import AgentNormalize
from shuffleAgent import AgentSplit
from grayAgent import AgentGray
from resizeAgent import AgentResize
import matplotlib.pyplot as plt
import os,shutil

class AgentInterface:
    def __init__(self):
        print("Iniciando pre processamento das imagens")
    def processamento(self, path, crop = True, cinza=False, size=(256,256,3), split=0.75, embaralhar = True):
        if crop:
            agente_crop = AgentCrop(size=size)
            dado_crop = agente_crop.crop_imagem(path)
        else:
            agente_res = AgentResize(size[0],size[1])
            dado_crop = agente_res.resize(path)
            
        if cinza:
            agente_cinza = AgentGray()
        
        agente_norm = AgentNormalize()
        agente_split = AgentSplit(split=split)
        
        
        if cinza:
            dado_cinza = agente_cinza.grayscale(dado_crop)
            dado_norm = agente_norm.normalizar_imagem(dado_cinza)
        else:
            dado_norm = agente_norm.normalizar_imagem(dado_crop)

        treino, teste = agente_split.dividir_dado(dado_norm, embaralhar)

        # Create folders with splited data
        base_dir = 'data_processed'
        base_name = 'image'

        train_dir = os.path.join(base_dir, 'train')
        test_dir = os.path.join(base_dir, 'test')
 
        if os.path.exists(train_dir):
            for filename in os.listdir(train_dir):
                file_path = os.path.join(train_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        
        if os.path.exists(test_dir):
            for filename in os.listdir(test_dir):
                file_path = os.path.join(test_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        counter_train = 0
        for image in treino:
            counter_train += 1
            caminho = os.path.join(train_dir, f'{base_name}_{counter_train}.png')
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(caminho, bbox_inches='tight', pad_inches=0)
            plt.close()

        print(f'Imagens de treino salva em {train_dir}')
        
        counter_test = 0
        for image in teste:
            counter_test += 1
            caminho = os.path.join(test_dir, f'{base_name}_{counter_test}.png')
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(caminho, bbox_inches='tight', pad_inches=0)
            plt.close()

        print(f'Imagens de teste salva em {test_dir}')
        
        return treino, teste