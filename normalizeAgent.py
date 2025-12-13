class AgentNormalize:
    def __init__(self):
        print("Agente Normalizador iniciado")
    def normalizar_imagem(self, dado):
        dado_norm = dado/255
        print(f"Imagem normalizada com sucesso. Intervalo de valores: [{dado_norm.min()}, {dado_norm.max()}]")
        return dado_norm