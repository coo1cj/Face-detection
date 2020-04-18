import cv2
import numpy as np

### Sliding window
def slide_win(x_straid,y_straid,img,window_size):
    """
        x_straid: step size in x direction
        y_straid: step size in y direction
        window_size: the size of sliding window
        return: tuple containing coordinates x,y and pixel matrix in sliding window
    """
    w = img.shape[1]
    h = img.shape[0]
    for i in range(1,h-window_size,y_straid):
        for j in range(1,w-window_size,x_straid):
            yield (j, i, img[i:i+window_size,j:j+window_size])


### Scaling (image pyramid)
def diff_image(scale,img):
    """
        scale: scaled size
        img：picture pixel matrix
        return: picture pixel matrix after scaling
    """
    yield img
    h = img.shape[0]
    w = img.shape[1]
    while h > 64 and w > 64:
        h /= scale
        w /= scale
        img = cv2.resize(img,(int(w),int(h)))
        yield img

### Non-maximum suppression
def nms(data):
    """
        data: the predicted result without non-maximum suppression
        return： the predicted result after non-maximum suppression
    """
    order = data[:,5].argsort()[::-1]
    delete = []
    x1 = data[:,2]
    y1 = data[:,1]
    x2 = data[:,4] + data[:,2]
    y2 = data[:,3] + data[:,1]
    ### Using the broadcast nature of numpy, caculate the area of all boxes
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  

    while order.size > 0:
        ### choose the index of the maximum area
        i = order[0]
        delete.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        ### Caculate the area of in intersecting box, replace the non-intersecting area with 0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ### Caculate the iou of all boxes and the maximum area box
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        ### choose the index of all boxes with the iou less than a threshold
        index = np.where(ovr <= 0.45)[0]   
        order = order[index + 1]
    return delete


### for show all the images of predicted faces in test
def showImg_test(data,path):
    '''
        data: the predicted results (txt)
        path: the path for saving the predicted images
    '''
    c = 0
    for i in range(1,501):
        img = cv2.imread('test/%04d.jpg'%i)
        ### select the prediction result of the corresponding picture
        a = data[data[:,0] == i]
        for j in range(len(a)):
            img1 = img[a[j,1]:a[j,1]+a[j,3],a[j,2]:a[j,2]+a[j,4]]
            cv2.imwrite(path + '/%05d.jpg'%c,img1)
            c += 1


### for show all the images of predicted faces in train
def showImg_test(data,path):
    '''
        data: the predicted results (txt)
        path: the path for saving the predicted images
    '''
    c = 0
    for i in range(1,1001):
        img = cv2.imread('train/%04d.jpg'%i)
        ### select the prediction result of the corresponding picture
        a = data[data[:,0] == i]
        for j in range(len(a)):
            img1 = img[a[j,1]:a[j,1]+a[j,3],a[j,2]:a[j,2]+a[j,4]]
            cv2.imwrite(path + '/%05d.jpg'%c,img1)
            c += 1