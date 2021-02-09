#adelia_data_set
import os
import numpy
from sklearn import tree
import cv2

features = []
labels = []

#1 = mascara on
#0 = mascara off
def read_image_for_ia(path, status):
    try:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (160, 160))
        img_array = numpy.array(img).flatten()
        features.append(img_array)
        labels.append(status)
    except:
        print("Imagem {} não foi alocada".format(path))
    
#maskon
path = "imagens/maskon/"
maskon = os.listdir(path)
for mask in maskon:
    read_image_for_ia(path + mask, 1)

#maskoff
path = "imagens/maskoff/"
maskoff = os.listdir(path)
for mask in maskoff:
    read_image_for_ia(path + mask, 0)

clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)