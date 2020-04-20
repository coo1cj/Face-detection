import numpy as np
from skimage.feature import hog
from skimage import io
import cv2
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import random as rand
import joblib


x = np.zeros((11284,3888))
y = np.hstack((np.ones(1284),np.zeros(10000)))


### Generate training set 
for i in range(1284):
    I = io.imread('train_pos/%04d.jpg'%i, as_gray=True)
    I = cv2.resize(I,(64,64))
    x[i,:] = hog(I,orientations=12,transform_sqrt=True)

for i in range(10000):
    I = io.imread('train_neg1/%05d.jpg'%i, as_gray=True)
    I = cv2.resize(I,(64,64))
    x[i+1284,:] = hog(I,orientations=12,transform_sqrt=True)


### Cross-validation 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = rand.randint(1, 100))

### See the result
clf = LinearSVC().fit(x_train,y_train)
print(clf.score(x_test,y_test))


### Training the classifier
clf = LinearSVC().fit(x,y)
print(clf.score(x_test,y_test))

### Save the classifier
joblib.dump(clf,'final_classifier/last_train_model_1.m')






