#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 09:02:56 2019

@author: helpthx
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from pylab import imread,imshow,figure,show,subplot
from numpy import reshape,uint8,flipud
from sklearn.cluster import MiniBatchKMeans
print('Versão da OpenCV: ', cv2.__version__, end='\n\n')

# Importando a imagem usada
img = cv2.imread('/home/helpthx/Documents/PROJETO_2/kiss512x512.jpg', cv2.IMREAD_GRAYSCALE)
plt.gray()
plt.imshow(img)
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/base512x512.png', img)

img.shape

# Função para fazer o sub sapling da imagem
def simple_subsampling(img, n):
  lista_imagens = []
  for i in range(n+1):
    img = img[1::2, 1::2]
    lista_imagens.append(img)
        
  return lista_imagens  

# Chamando a função
lista_imagens = simple_subsampling(img, 4)

lista_imagens[0].shape
# (256, 256)

lista_imagens[1].shape
# (128, 128)

lista_imagens[2].shape
# (64, 64)

lista_imagens[3].shape
# (32, 32)

lista_imagens[4].shape
# (16, 16)


# Salvando imagens 
plt.imshow(lista_imagens[0])
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/sub_samplu256x.png', 
           lista_imagens[0])


plt.imshow(lista_imagens[1])
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/sub_samplu128x.png', 
           lista_imagens[1])


plt.imshow(lista_imagens[2])
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/sub_samplu64x.png', 
           lista_imagens[2])


plt.imshow(lista_imagens[3])
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/sub_samplu32x.png', 
           lista_imagens[3])

# Interpolação nn
def nn_interpolate(A, new_size):
    """Vectorized Nearest Neighbor Interpolation"""

    old_size = A.shape
    row_ratio, col_ratio = np.array(new_size)/np.array(old_size)

    # row wise interpolation 
    row_idx = (np.ceil(range(1, 1 + int(old_size[0]*row_ratio))/row_ratio) - 1).astype(int)

    # column wise interpolation
    col_idx = (np.ceil(range(1, 1 + int(old_size[1]*col_ratio))/col_ratio) - 1).astype(int)

    final_matrix = A[:, row_idx][col_idx, :]

    return final_matrix

# Chamando a função
nn_iterpolation256 = nn_interpolate(lista_imagens[0], 512)
nn_iterpolation128 = nn_interpolate(lista_imagens[1], 512)
nn_iterpolation64 = nn_interpolate(lista_imagens[2], 512)
nn_iterpolation32 = nn_interpolate(lista_imagens[3], 512)

# Salvando arquivos
plt.imshow(nn_iterpolation256)
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/nn_inteporlation256x.png', 
           nn_iterpolation256)


plt.imshow(nn_iterpolation128)
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/nn_inteporlation128x.png', 
           nn_iterpolation128)


plt.imshow(nn_iterpolation64)
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/nn_inteporlation64x.png', 
           nn_iterpolation64)


plt.imshow(nn_iterpolation32)
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/nn_inteporlation32x.png', 
           nn_iterpolation32)

# Interpolação Biliniar
# Bilinear interpolation
def bilinear_interpolate(image):
  image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
  (h, w, channels) = image.shape
  
  h2 = 512
  w2 = 512
  temp = np.zeros((h2, w2, 3), np.uint8)
  x_ratio = float((w - 1)) / w2;
  y_ratio = float((h - 1)) / h2;
  for i in range(1, h2 - 1):
    for j in range(1 ,w2 - 1):
      x = int(x_ratio * j)
      y = int(y_ratio * i)
      x_diff = (x_ratio * j) - x
      y_diff = (y_ratio * i) - y
      a = image[x, y] & 0xFF
      b = image[x + 1, y] & 0xFF
      c = image[x, y + 1] & 0xFF
      d = image[x + 1, y + 1] & 0xFF
      blue = a[0] * (1 - x_diff) * (1 - y_diff) + b[0] * (x_diff) * (1-y_diff) + c[0] * y_diff * (1 - x_diff)   + d[0] * (x_diff * y_diff)
      green = a[1] * (1 - x_diff) * (1 - y_diff) + b[1] * (x_diff) * (1-y_diff) + c[1] * y_diff * (1 - x_diff)   + d[1] * (x_diff * y_diff)
      red = a[2] * (1 - x_diff) * (1 - y_diff) + b[2] * (x_diff) * (1-y_diff) + c[2] * y_diff * (1 - x_diff)   + d[2] * (x_diff * y_diff)
      temp[j, i] = (blue, green, red)

  return cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

# Chamando as funções
bl_iterpolation256 = bilinear_interpolate(lista_imagens[0])
bl_iterpolation128 = bilinear_interpolate(lista_imagens[1])
bl_iterpolation64 = bilinear_interpolate(lista_imagens[2])
bl_iterpolation32 = bilinear_interpolate(lista_imagens[3])

# Salvando arquivos
plt.imshow(bl_iterpolation256)
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/bl_inteporlation256x.png', 
           bl_iterpolation256)


plt.imshow(bl_iterpolation128)
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/bl_inteporlation128x.png', 
           bl_iterpolation128)


plt.imshow(bl_iterpolation64)
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/bl_inteporlation64x.png', 
           bl_iterpolation64)


plt.imshow(bl_iterpolation32)
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/bl_inteporlation32x.png', 
           bl_iterpolation32)


# Qubatização de niveis de cinza
image = cv2.imread('/home/helpthx/Documents/PROJETO_2/kiss512x512.jpg', 
                   cv2.IMREAD_GRAYSCALE)
print(image.shape)

def quantizador_kmeans(image, n):
    # Extract width & height of image
    (HEIGHT, WIDTH) = image.shape[:2]
    
    # Convert image to L, A, B color space
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Reshape the image to a feature vector
    image = image.reshape((image.shape[0] * image.shape[1], 1))
    
    # Apply MiniBatchKMeans and then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = n)
    labels = clt.fit_predict(image)
    print(labels)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    
    # reshape the feature vectors to images
    quant = quant.reshape((HEIGHT, WIDTH))
    image = image.reshape((HEIGHT, WIDTH))
    
    return quant, image

quant_8, image_8 = quantizador_kmeans(image, 8)
quant_4, image_4 = quantizador_kmeans(image, 4)
quant_2, image_2 = quantizador_kmeans(image, 2)
quant_1, image_1 = quantizador_kmeans(image, 1)

# Salvando arquivos
plt.gray()

plt.imshow(quant_8)
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/quant_8.png', 
           quant_8)


plt.imshow(quant_4)
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/quant_4.png', 
           quant_4)


plt.imshow(quant_2)
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/quant_2.png', 
           quant_2)


plt.imshow(quant_1)
plt.show()
plt.imsave('/home/helpthx/Documents/PROJETO_2/quant_1.png', 
           quant_1)

