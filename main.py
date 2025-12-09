import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

from gaussFilterAgent import Agente_Filtro_Gauss
from interfaceAgent import Agente_Interface

def main():
    parser = argparse.ArgumentParser(description="Processamento de imagens com agentes.")

    # 2. Adiciona os argumentos esperados
    # Argumento para 'path' (Posicional e obrigatório)
    parser.add_argument('path', type=str, help='Caminho para o diretório de imagens.')

    # Argumento para 'crop' (Flag booleana)
    parser.add_argument('--crop', action=argparse.BooleanOptionalAction, default=True,
                        help='Se deve aplicar recorte (crop) ou redimensionamento (resize). Padrão: True.')

    # Argumento para 'cinza' (Flag booleana)
    parser.add_argument('--cinza', action=argparse.BooleanOptionalAction, default=False,
                        help='Se deve converter para tons de cinza. Padrão: False.')

    # Argumento para 'size' (String que será convertida)
    parser.add_argument('--size', type=str, default='256,256,3',
                        help='Dimensões de saída (width,height,channels). Ex: "500,500,3".')

    # Argumento para 'split' (Float)
    parser.add_argument('--split', type=float, default=0.75,
                        help='Proporção de treino (0.0 < split < 1.0). Padrão: 0.75.')

    # Argumento para 'embaralhar' (Flag booleana)
    parser.add_argument('--embaralhar', action=argparse.BooleanOptionalAction, default=True,
                        help='Se deve embaralhar os dados antes de dividir. Padrão: True.')

    # 3. Analisa os argumentos da linha de comando
    args = parser.parse_args()

    # 4. Converte a string 'size' para uma tupla de inteiros
    try:
        size_list = [int(s.strip()) for s in args.size.split(',')]
        if len(size_list) != 3:
            raise ValueError("O argumento --size deve ter 3 valores separados por vírgula (width,height,channels).")
        size_tuple = tuple(size_list)
    except ValueError as e:
        print(f"Erro ao analisar o argumento --size: {e}")
        return # Encerra o programa em caso de erro

    # 5. Chama a lógica principal com os argumentos
    interface = Agente_Interface()
    treino, teste = interface.processamento(
        path=args.path,
        crop=args.crop,
        cinza=args.cinza,
        size=size_tuple,  # Passa a tupla convertida
        split=args.split,
        embaralhar=args.embaralhar
    )
    return treino, teste
    
if __name__ == "__main__":
    treino, teste = main()
    
    plt.imshow(treino[10], cmap='gray')
    plt.show()

    filtro_gauss = Agente_Filtro_Gauss()
    treino_filtrado = filtro_gauss.filtragem(treino, 5)

    plt.imshow(treino_filtrado[10], cmap='gray')
    plt.show()  