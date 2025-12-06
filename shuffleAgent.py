import numpy as np

class Agente_Divisor:
    def __init__(self, split = 0.75):
        if not 0.0 < split < 1.0:
             raise ValueError("A proporção deve ser um valor maior que 0 e menor que 1.")
        self.split = split
        print(f"Agente de Divisão inicializado. Proporção de Treinamento: {self.split* 100:.0f}%")
    def dividir_dado(self, dado, embaralhar = True):
        div = int(dado.shape[0]*self.split)
        dados_embaralhados = dado.copy()
        if embaralhar:
            np.random.shuffle(dados_embaralhados)
        x_train = dados_embaralhados[:div,:,:]
        x_test = dados_embaralhados[div:,:,:]
        print("Divisão concluida em conjunto de teste e treino.")
        return x_train, x_test