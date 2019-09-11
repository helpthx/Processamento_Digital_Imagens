from PIL import Image, ImageDraw
import sys
import math, random
from itertools import product

class UFarray:
    def __init__(self):
        self.P = []
        self.label = 0

    def makeLabel(self):
        r = self.label
        self.label += 1
        self.P.append(r)
        return r
    
    def setRoot(self, i, root):
        while self.P[i] < i:
            j = self.P[i]
            self.P[i] = root
            i = j
        self.P[i] = root

    def findRoot(self, i):
        while self.P[i] < i:
            i = self.P[i]
        return i
    

    def find(self, i):
        root = self.findRoot(i)
        self.setRoot(i, root)
        return root
    
    def union(self, i, j):
        if i != j:
            root = self.findRoot(i)
            rootj = self.findRoot(j)
            if root > rootj: root = rootj
            self.setRoot(j, root)
            self.setRoot(i, root)
    
    def flatten(self):
        for i in range(1, len(self.P)):
            self.P[i] = self.P[self.P[i]]
    
    def flattenL(self):
        k = 1
        for i in range(1, len(self.P)):
            if self.P[i] < i:
                self.P[i] = self.P[self.P[i]]
            else:
                self.P[i] = k
                k += 1
                
def run(img):
    data = img.load()
    width, height = img.size

    uf = UFarray()

    labels = {}
 
    for y, x in product(range(height), range(width)):
 
        if data[x, y] == 255:
            pass
 
        elif y > 0 and data[x, y-1] == 0:
            labels[x, y] = labels[(x, y-1)]
 
        elif x+1 < width and y > 0 and data[x+1, y-1] == 0:
 
            c = labels[(x+1, y-1)]
            labels[x, y] = c

            if x > 0 and data[x-1, y-1] == 0:
                a = labels[(x-1, y-1)]
                uf.union(c, a)
 
            elif x > 0 and data[x-1, y] == 0:
                d = labels[(x-1, y)]
                uf.union(c, d)
 
        elif x > 0 and y > 0 and data[x-1, y-1] == 0:
            labels[x, y] = labels[(x-1, y-1)]
 
        elif x > 0 and data[x-1, y] == 0:
            labels[x, y] = labels[(x-1, y)]
 
        else: 
            labels[x, y] = uf.makeLabel()
 
    uf.flatten()
 
    colors = {}

    output_img = Image.new("RGB", (width, height))
    outdata = output_img.load()

    for (x, y) in labels:
 
        component = uf.find(labels[(x, y)])

        labels[(x, y)] = component
 
        if component not in colors: 
            colors[component] = (random.randint(0,255), random.randint(0,255),random.randint(0,255))

        outdata[x, y] = colors[component]

return (labels, output_img)

# Inicializar a imagem
img = Image.open('/content/fig4.jpg')
img.show()

img = img.point(lambda p: p > 190 and 255)
img = img.convert('1')

(labels, output_img) = run(img)

output_img.show()

output_img

type(output_img)

type(output_img)

pix.shape

uniqueValues = np.unique(pix)
uniqueValues

import cv2
import matplotlib.pyplot as plt
gray = cv2.cvtColor(pix, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)

gray.shape

uniqueValues = np.unique(gray)
uniqueValues

len(uniqueValues) - 1

# Open the image
img = Image.open('/content/fig4.jpg')
img.show()
# img = np.array(img)
# img = cv2.bitwise_not(img)
# img = Image.fromarray(img)

# Threshold the image, this implementation is designed to process b+w
# images only
img = img.point(lambda p: p <= 190 and 255)
img = img.convert('1')

# labels is a dictionary of the connected component data in the form:
#     (x_coordinate, y_coordinate) : component_id
#
# if you plan on processing the component data, this is probably what you
# will want to use
#
# output_image is just a frivolous way to visualize the components.
(labels, output_img_holes) = run(img)

output_img_holes.show()

output_img_holes

import numpy as np
import cv2
import matplotlib.pyplot as plt

pix_holes = np.array(output_img_holes)
gray_holes = cv2.cvtColor(pix_holes, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_holes)

uniqueValues_holes = np.unique(gray_holes)
len(uniqueValues_holes) - 2

new_img = cv2.addWeighted(gray_holes, 1, gray, 1, 0)
plt.imshow(new_img) 

