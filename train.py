import numpy as np
from skimage.feature import hog
from skimage import io
import cv2
from skimage.color import rgb2gray
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import random as rand
import joblib
from util import *

### load the first classifier
clf = joblib.load('final_classifier/last_train_model_1.m')


### choose the parameters
window_size = 96
scale = 1.3
x_straid = 12
y_straid = 12
box = np.zeros(6)

### run the sliding window with the scaling
for i in range(1,1001):
    image = cv2.imread('train/%04d.jpg'%i)
    count = 1
    for pic in diff_image(scale,image):
        pic1 = np.copy(pic)
        pic1 = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)
        for (x, y, img) in slide_win(x_straid,y_straid,pic,window_size):
            img = cv2.resize(img,(64,64))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            feature = hog(img,orientations=12,transform_sqrt=True).reshape(1,-1)#pic1[y:y+window_size,x:x+window_size]
            if clf.predict(feature) == 1:
                #cv2.rectangle(pic2,(x,y),(x+window_size,y+window_size),(0,0,255))
                #cv2.imwrite('train_pre/%05d.jpg'%c,pic[int(y*count):int(y*count+window_size*count),int(x*count):int(x*count+window_size*count)])
                #c +=1
                ### Make the size of intercepted window consistent with the original image
                box = np.vstack((box,[i,y*count,x*count,window_size*count,window_size*count,max(clf.decision_function(feature))]))
        count *= scale

### save the training result of prediction 
box = np.array(box)
save_txt('train_predict.txt',box)


### The following is selecting the falsepositive data
res = np.zeros(6)

### it's a txt after adding the scores (100.000) to the label_train
d1 = np.loadtxt('change.txt')

### Combining predicted and correct data
for i in range(1,1001):
    t = d1[d1[:,0] == i]
    b = box[box[:,0] == i]
    res = np.vstack((res,t))
    res = np.vstack((res,b))

### Using the function nms for removing the truepositive
res1 = np.zeros(6)
for i in range(1,1001):
    d = res[res[:,0] == i]
    res1 = np.vstack((res1,d[nms(d)]))
res1 = res1[res1[:,5] != 100.000]

### Save the falsepositive
save_txt('falsepotive.txt',res1[1:])