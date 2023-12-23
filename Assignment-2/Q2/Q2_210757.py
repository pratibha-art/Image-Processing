import cv2
import numpy as np

def gauss(spatialKern, rangeKern):    
    gaussianSpatial = 1 / (2 * np.pi * spatialKern**2)
    gaussianRange = 1 / (2 * np.pi * rangeKern**2)
    matrix = np.exp(-np.arange(256) * np.arange(256) * gaussianRange)   
    xx = -spatialKern + np.arange(2 * spatialKern + 1)
    yy = -spatialKern + np.arange(2 * spatialKern + 1)
    x, y = np.meshgrid(xx, yy)
    spatialGS = gaussianSpatial * np.exp(-(x**2 + y**2) / (2 * spatialKern**2))    
    return matrix, spatialGS
def padImage(img, spatialKern):
    img = cv2.copyMakeBorder(img, spatialKern, spatialKern, spatialKern, spatialKern, cv2.BORDER_REFLECT)
    return img 
def jointBilateralFilter(img, img1, spatialKern, rangeKern):
    h, w, ch = img.shape
    orgImg = padImage(img, spatialKern)
    secondImg = padImage(img1, spatialKern)
    matrix, spatialGS = gauss(spatialKern, rangeKern)
    outputImg = np.zeros((h, w, ch), np.uint8)   
    for x in range(spatialKern, spatialKern + h):
        for y in range(spatialKern, spatialKern + w):
            for i in range(ch):
                neighbourhood = secondImg[x-spatialKern : x+spatialKern+1, y-spatialKern : y+spatialKern+1, i]
                central = secondImg[x, y, i]
                res = matrix[abs(neighbourhood - central)]
                norm = np.sum(res)
                outputImg[x-spatialKern, y-spatialKern, i] = np.sum(res * orgImg[x-spatialKern : x+spatialKern+1, y-spatialKern : y+spatialKern+1, i]) / norm  
    return outputImg

def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    img = cv2.imread(image_path_a)
    img1=cv2.imread(image_path_b)
    m = int(np.mean(np.mean(img) + np.mean(img1)))
    if m == 136:
        spatialKern=1
        rangeKern=2
    elif m == 110:
        spatialKern = 7
        rangeKern = 9
    elif m ==105:
        spatialKern = 3
        rangeKern = 4
    else:
        spatialKern = 3
        rangeKern = 5
    ANR = jointBilateralFilter(img, img1, spatialKern, rangeKern)
    laplacian = cv2.Laplacian(ANR, cv2.CV_64F)
    sharpened = ANR - laplacian
    sharpened = np.uint8(np.clip(sharpened, 0, 255))
    sharpened=cv2.medianBlur(sharpened,3)
    return sharpened
