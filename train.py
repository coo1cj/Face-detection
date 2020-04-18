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




clf = joblib.load('train_model_first.m')


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






#np.savetxt('train_predict_2.txt',box,fmt='%d %d %d %d %d %.3f')  


""" cropImg = img
                cv2.imwrite('fp/%05d.jpg'%count,cropImg) 
                count += 1 """

        #train[:,1:3] /= 1.5
"""               for j in range(len(train[train[:,0] == i])):
                    if IoU(a,train[train[:,0] == i][j,1:]) < 0.15: """
"""     im = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
    cv2.imshow("ff",image)
    cv2.waitKey(0) """
      
 
""" for i in range(1,10):
    image = cv2.imread('train/%04d.jpg'%i)
    
    for pic in diff_image(scale,image):
        pic1 = np.copy(pic)
        #pic2 = np.copy(pic)
        pic1 = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)
        for (x, y, img) in slide_win(8,8,pic,window_size):
            feature = hog(pic1[y:y+window_size,x:x+window_size]).reshape(1,-1)
            if clf.predict(feature) == 1:
                a = [y,x,64,64]
                for j in range(len(train[train[:,0] == i])):
                    if IoU(a,train[train[:,0] == i][j,1:]) < 0.5:
                        cropImg = img
                        cv2.imwrite('false_pos/%03d.jpg'%count,cropImg)   
                        count += 1  
        train[:,1:3] /= scale """