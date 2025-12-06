import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from gaussFilterAgent import Agente_Filtro_Gauss
from interfaceAgent import Agente_Interface


folder_dir = 'imagens\DIV2K_train_HR'
interface = Agente_Interface(folder_dir)

treino, teste = interface.processamento(folder_dir, cinza=True, size=(500,500,3))

plt.imshow(treino[0], cmap='gray')
plt.show()

filtro_gauss = Agente_Filtro_Gauss()
treino_filtrado = filtro_gauss.filtragem(treino, 5)

plt.imshow(treino_filtrado[0], cmap='gray')
plt.show()