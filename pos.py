from PIL import Image
import numpy as np


### pour gerer les images positives
train = np.loadtxt('label_train.txt')
k = 0
for i in range(1,1001):
    im = Image.open('train/%04d.jpg'%i)
    for j in range(len(train[train[:,0] == i])):
        y, x, h, w = train[train[:,0] == i][j,1:]
        region = im.crop((x, y, x+w, y+h))
        region.save('train_pos/%04d.jpg'%k)
        k += 1

