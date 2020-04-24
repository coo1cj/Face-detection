import numpy as np
from skimage.feature import hog
from skimage import io
import cv2
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import random as rand
import joblib


### load the truepositive and falsepositive samples
fp = np.loadtxt('falsepositive1.txt',dtype=int)
ros = np.loadtxt('ros.txt',dtype=int)
fp1 = np.loadtxt('falsepositive2.txt',dtype=int)


x = np.zeros((15307,3888))
y = np.hstack((np.zeros(12484),np.ones(2823)))

### add the negative samples
count = 0
for i in range(1,1001):
    I = io.imread('train/%04d.jpg'%i,as_gray=True)
    a = fp[fp[:,0] == i]

    for j in range(len(a)):
        yy = a[j,1]
        xx = a[j,2]
        hh = a[j,3]
        ww = a[j,4]
        f = I[yy:yy+hh,xx:xx+ww]
        f = cv2.resize(f,(64,64))
        x[count,:] = hog(f,orientations=12,transform_sqrt=True)
        count += 1

for i in range(10000):
    I = io.imread('train_neg1/%05d.jpg'%i, as_gray=True)
    I = cv2.resize(I,(64,64))
    x[i+1871,:] = hog(I,orientations=12,transform_sqrt=True)


count1 = 0
for i in range(1,1001):
    I = io.imread('train/%04d.jpg'%i,as_gray=True)
    a = fp1[fp1[:,0] == i]

    for j in range(len(a)):
        yy = a[j,1]
        xx = a[j,2]
        hh = a[j,3]
        ww = a[j,4]
        f = I[yy:yy+hh,xx:xx+ww]
        f = cv2.resize(f,(64,64))
        x[count1+11871,:] = hog(f,orientations=12,transform_sqrt=True)
        count1 += 1



### add the positive samples
for i in range(1284):
    I = io.imread('train_pos/%04d.jpg'%i, as_gray=True)
    I = cv2.resize(I,(64,64))
    x[i+12484,:] = hog(I,orientations=12,transform_sqrt=True)

for i in range(1284):
    I = io.imread('apresrotation/%04d.jpg'%i, as_gray=True)
    I = cv2.resize(I,(64,64))
    x[i+13768,:] = hog(I,orientations=12,transform_sqrt=True)   

c1 = 0
for i in range(1,1001):
    I = io.imread('train/%04d.jpg'%i,as_gray=True)
    a = ros[ros[:,0] == i]

    for j in range(len(a)):
        yy = a[j,1]
        xx = a[j,2]
        hh = a[j,3]
        ww = a[j,4]
        f = I[yy:yy+hh,xx:xx+ww]
        f = cv2.resize(f,(64,64))
        x[c1+15052,:] = hog(f,orientations=12,transform_sqrt=True)
        c1 += 1



### cross-validation
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = rand.randint(1, 100))

### try the different paremeters
""" C = np.linspace(0.01,3)
dual = [True,False]
losses=['hinge','squared_hinge']
c_w = ['balanced',None]

m = 0
for c in C:
    for l in losses:
        for cw in c_w:
            clf = LinearSVC(C=c,loss=l,class_weight=cw).fit(x_train,y_train)
            if m < clf.score(x_test,y_test):
                m = clf.score(x_test,y_test)
                l_res = l
                cw_res = cw
                c_res = c
print(m)
print(l_res)
print(cw_res)
print(c_res) """



### training the classifier again
clf = LinearSVC(C=0.2,loss='hinge').fit(x,y)

### save new classifier
joblib.dump(clf,'final_classifier/train_model_2.m')
