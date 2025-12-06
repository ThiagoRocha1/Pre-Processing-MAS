import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import gaussFilterAgent as Agente_Filtro_Gauss
import interfaceAgent as Agente_Interface


interface = Agente_Interface()
treino, teste = interface.processamento(folder_dir, cinza=True, size=(500,500,3))

plt.imshow(treino[10], cmap='gray')
plt.show()

filtro_gauss = Agente_Filtro_Gauss()
treino_filtrado = filtro_gauss.filtragem(treino, 5)

plt.imshow(treino_filtrado[10], cmap='gray')
plt.show()