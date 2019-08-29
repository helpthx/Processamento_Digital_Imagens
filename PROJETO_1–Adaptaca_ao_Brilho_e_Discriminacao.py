import cv2

print('Versão da OpenCV: ', cv2.__version__, end='\n\n')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

qtdeLinhas = 1023
qtdeColunas = 1023

W = np.zeros((qtdeLinhas, qtdeColunas, 3))

# plt.figure(figsize=(10, 10))
# plt.imshow(W)

intensidade_pixel = 0
OffSetLinha = 1
OffSetColuna = 1
niveis_cinza_lista = []
sair = 1

while (sair):
    plt.figure(figsize=(10, 10))
    plt.imshow(W)
    plt.pause(0.5)
    input_controle = input(
        '\nDigite 1 para incrementar e 0 para decrementar:\nDigite 3 para ir para o proximo quadrado, Digite 4 para voltar para o quadrado anterior\n Digite sair para sair salvar e Digite 5 para salvar o documento atual ')
    plt.close()
    if input_controle == "1":
        intensidade_pixel = intensidade_pixel + 1
        for i in range(8 * OffSetLinha, qtdeLinhas - 7 * OffSetLinha):
            for j in range(8 * OffSetColuna, qtdeColunas - 7 * OffSetColuna):
                W[i][j] = intensidade_pixel / 255

    elif input_controle == "0":
        intensidade_pixel = intensidade_pixel - 1
        for i in range(8, qtdeLinhas - 7):
            for j in range(8, qtdeColunas - 7):
                W[i][j] = intensidade_pixel / 255

        # plt.figure(figsize=(10, 10))
        # plt.imshow(W)
        #plt.pause(0.5)

    elif input_controle == "3":
        niveis_cinza_lista.append(intensidade_pixel)
        OffSetLinha = OffSetLinha + 1
        OffSetColuna = OffSetColuna + 1

        # plt.figure(figsize=(10, 10))
        # plt.imshow(W)
        # plt.pause(0.5)

    elif input_controle == "4":
        OffSetLinha = OffSetLinha - 1
        OffSetColuna = OffSetColuna - 1

        # plt.figure(figsize=(10, 10))
        # plt.imshow(W)
        # plt.pause(0.5)

    elif input_controle == "5":
        plt.close()
        plt.figure(figsize=(10, 10))
        plt.imshow(W)
        plt.savefig("quadrado_niveis_cinza.png")
        plt.show()
        plt.pause(0.5)

    elif input_controle == "sair":
        plt.close()
        plt.figure(figsize=(10, 10))
        plt.imshow(W)
        plt.savefig("quadrado_niveis_cinza.png")
        plt.show()
        sair = 0

dt = pd.DataFrame(niveis_cinza_lista)
fig = dt.plot(figsize=(20, 16)).get_figure()
fig.savefig('grafico_niveis_cinza', dpi=300)
indices = []
x = 0
for i in enumerate(niveis_cinza_lista):
    indices.append(x)
    x = x + 1

z = np.polyfit(niveis_cinza_lista, indices, 2)



