import cv2
import random
import numpy as np
from util import IoU



### Generate negative samples(here i choose the size of all the negative samples is 96x96)
def generate_neg(num,nb): 
    """
        num: the number of the picture
        nb: the number of saved image
    """
    img=cv2.imread('train/%04d.jpg'%num)
    d = np.loadtxt('label_train.txt')
    while 1:   
        count = 0
        ### randomly generate coordinates x,y
        y = random.randint(0, img.shape[0]-96)
        x = random.randint(0, img.shape[1]-96)
        a = [y, x, 96, 96]
        for i in range(len(d[d[:,0] == num])):
            if IoU(a,d[d[:,0] == num][i,1:]) < 0.05:
                count += 1
            else:
                break
        if count == len(d[d[:,0] == num]):
            break
    
    cropImg = img[(y):(y + 96), (x):(x + 96)]
    cv2.imwrite('train_neg1/%05d.jpg'%nb, cropImg)


### Generate 10 negative samples per image
k = 0
for i in range(1,1001):
    for j in range(k*10,k*10+10):
        generate_neg(i,j)
    k += 1