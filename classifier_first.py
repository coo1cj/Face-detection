import numpy as np
from skimage.feature import hog
from skimage import io
import cv2
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import random as rand
import joblib
from sklearn import metrics


x = np.zeros((12568,3888))
y = np.hstack((np.ones(2568),np.zeros(10000)))


### Generate training set 
for i in range(1284):
    I = io.imread('train_pos/%04d.jpg'%i, as_gray=True)
    I = cv2.resize(I,(64,64))
    x[i,:] = hog(I,orientations=12,transform_sqrt=True)


for i in range(1284):
    I = io.imread('apresrotation/%04d.jpg'%i, as_gray=True)
    I = cv2.resize(I,(64,64))
    x[i+1284,:] = hog(I,orientations=12,transform_sqrt=True)


for i in range(10000):
    I = io.imread('train_neg1/%05d.jpg'%i, as_gray=True)
    I = cv2.resize(I,(64,64))
    x[i+2568,:] = hog(I,orientations=12,transform_sqrt=True)


### Cross-validation 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

### try the different paremeters
""" C = np.linspace(0.01,1,num=30)
losses=['hinge','squared_hinge']
c_w = ['balanced',None]

m = 0
for c in C:
    for l in losses:
        for cw in c_w:
            clf = LinearSVC(C=c,loss=l,class_weight=cw,max_iter=2000).fit(x_train,y_train)
            if m < metrics.f1_score(y_test,clf.predict(x_test)):
                m = metrics.f1_score(y_test,clf.predict(x_test))
                l_res = l
                cw_res = cw
                c_res = c
print(m)
print(l_res)
print(cw_res)
print(c_res) """


### Training the classifier with the best parameters
clf1 = LinearSVC(C=0.0782,loss='squared_hinge',class_weight='balanced').fit(x,y)

### Save the classifier
joblib.dump(clf1,'final_classifier/train_model_1.m')






