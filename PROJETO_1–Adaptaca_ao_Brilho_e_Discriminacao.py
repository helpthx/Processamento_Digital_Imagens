# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

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
deltaintensidade_pixel = 0
OffSetLinha = 1
OffSetColuna = 1
niveis_cinza_lista = []
deltaintensidade_list = []
sair = 1

while (sair):
    plt.figure(figsize=(10, 10))
    plt.imshow(W)
    plt.pause(0.5)
    input_controle = input(
        '\nDigite 1 para incrementar e 0 para decrementar: \
        \nDigite 3 para ir para o proximo quadrado, \
        Digite 4 para voltar para o quadrado anterior\
        \n Digite sair para sair salvar e Digite 5 para salvar o \
        documento atual ')
    plt.close()
    if input_controle == "1":
        intensidade_pixel = intensidade_pixel + 1
        deltaintensidade_pixel = deltaintensidade_pixel + 1 
        for i in range(8 * OffSetLinha, qtdeLinhas - 7 * OffSetLinha):
            for j in range(8 * OffSetColuna, qtdeColunas - 7 * OffSetColuna):
                W[i][j] = intensidade_pixel / 255

    elif input_controle == "0":
        intensidade_pixel = intensidade_pixel - 1
        deltaintensidade_pixel = deltaintensidade_pixel + 1
        for i in range(8, qtdeLinhas - 7):
            for j in range(8, qtdeColunas - 7):
                W[i][j] = intensidade_pixel / 255

        # plt.figure(figsize=(10, 10))
        # plt.imshow(W)
        #plt.pause(0.5)

    elif input_controle == "3":
        niveis_cinza_lista.append(intensidade_pixel)
        deltaintensidade_list.append(deltaintensidade_pixel)
        OffSetLinha = OffSetLinha + 1
        OffSetColuna = OffSetColuna + 1
        deltaintensidade_pixel = 0

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
        plt.show()
        plt.pause(0.5)
        plt.savefig("quadrado_niveis_cinza.png")


    elif input_controle == "sair":
        plt.close()
        plt.figure(figsize=(10, 10))
        plt.imshow(W)
        plt.show()
        plt.pause(0.5)
        plt.savefig("quadrado_niveis_cinza.png")
        plt.show()
        sair = 0


print('Lista de niveis de cinza', niveis_cinza_lista)
print('Lista de delta niveis de cinza', deltaintensidade_list)

dt = pd.DataFrame(niveis_cinza_lista)
fig = dt.plot(figsize=(16, 16), legend=False)
fig.set_ylabel('Nivel de Pixel')
fig.set_xlabel('Incremento ao longo do tempo')
fig.set_title('Intensidade de Pixel ao longo do tempo')
fig.get_figure()
fig.savefig('grafico_niveis_cinza', dpi=300)
indices = []

df = pd.DataFrame(deltaintensidade_list)
fig1 = df.plot(figsize=(16, 16), legend=False)
fig1.set_ylabel('Delta da intensidade')
fig1.set_xlabel('Incremento ao longo do tempo')
fig1.set_title('Delta da intensidade de Pixel ao longo do tempo')
fig1.get_figure()
fig1.savefig('grafico_delta_niveis_cinza', dpi=300)

x = 0
indices_1 = []
for i in enumerate(niveis_cinza_lista):
    indices_1.append(x)
    x = x + 1

poly_deg = 10 #degree of the polynomial fit
polynomial_fit_coeff_1 = np.polyfit(niveis_cinza_lista, indices_1, poly_deg)

start = 0
stop = 255
num_points = 255
arbitrary_time = np.linspace(start, stop, num_points)

lon_intrp_2 = np.polyval(polynomial_fit_coeff_1, 
                         arbitrary_time)

plt.plot(arbitrary_time, lon_intrp_2, 'r') #interpolated window as a red curve



x = 0
indices_2 = []
for i in enumerate(deltaintensidade_list):
    indices_2.append(x)
    x = x + 1

polynomial_fit_coeff_2 = np.polyfit(deltaintensidade_list, indices_2, 
                                    poly_deg)

lon_intrp_2 = np.polyval(polynomial_fit_coeff_2, 
                         arbitrary_time)

plt.plot(arbitrary_time, lon_intrp_2, 'r') #interpolated window as a red curve
