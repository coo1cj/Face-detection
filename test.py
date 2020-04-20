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

### load the second classifier
clf = joblib.load('final_classifier/last_train_model_2.m')

### choose the parameters
window_size = 96
scale = 1.4
x_straid = 10
y_straid = 10
box = np.zeros(6)

### run the sliding window with the scaling
for i in range(1,501):
    image = cv2.imread('test/%04d.jpg'%i)
    count = 1
    for pic in diff_image(scale,image):
        pic1 = np.copy(pic)
        pic1 = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)
        for (x, y, img) in slide_win(x_straid,y_straid,pic,window_size):
            img = cv2.resize(img,(64,64))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            feature = hog(img,orientations=12,transform_sqrt=True).reshape(1,-1)
            if clf.predict(feature) == 1:
                #img1 = image[int(y*count):int(y*count+window_size*count), int(x*count):int(x*count+window_size*count)]
                #cv2.rectangle(image,(x,y),(x*count+window_size*count,y*count+window_size*count),(0,0,255))

                ### Make the size of intercepted window consistent with the original image
                box = np.vstack((box,[i,y*count,x*count,window_size*count,window_size*count,max(clf.decision_function(feature))]))
                #cv2.imwrite('win96_train_fp/%05d.jpg'%c,img1)
                #c += 1
        count *= scale
box = np.array(box)

### use the function non-maximum suppression
res = np.zeros(6)
for i in range(1,501):
    d = box[box[:,0] == i]
    res = np.vstack((res,d[nms(d)]))

### filter out the confidence scores greater than 0.7
#res = res[res[:,5] > 0.7]

### save the final prediction
save_txt('test_predict_419.txt',res)






    




