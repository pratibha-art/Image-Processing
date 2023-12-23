import cv2
import numpy as np

# Usage
def add_padding(image, top, bottom, left, right):
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image
def threshold_image(input_image):
    _, thresholded_image = cv2.threshold(input_image, 1, 255, cv2.THRESH_BINARY)
    return thresholded_image
def find_corners(thresholded_image):
    corners = cv2.goodFeaturesToTrack(thresholded_image, maxCorners=4, qualityLevel=0.01, minDistance=10)
    corners = np.int0(corners)
    return corners
def sort_corners_clockwise(corners):
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:,0,1] - center[0,1], corners[:,0,0] - center[0,0])
    sorted_corners = corners[np.argsort(angles)]
    return sorted_corners
def perspective_transform(image, sorted_corners):
    target_size = (600, 600)
    target_corners = np.array([[0, 0], [target_size[0]-1, 0], [target_size[0]-1, target_size[1]-1], [0, target_size[1]-1]], dtype='float32')
    sorted_corners = np.array(sorted_corners, dtype='float32')
    M = cv2.getPerspectiveTransform(sorted_corners, target_corners)
    warped_image = cv2.warpPerspective(image, M, target_size)
    return warped_image
def solution(image_path):
    image= cv2.imread(image_path)
    h,w,_ = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    top_padding = 50
    bottom_padding = 50
    left_padding = 50
    right_padding = 50
    padded_image = add_padding(gray_image, top_padding, bottom_padding, left_padding, right_padding)
    thresholded_image = threshold_image(padded_image)
    blurred_image = cv2.GaussianBlur(thresholded_image, (5, 5), 0)
    corners = find_corners(blurred_image)
    unpadded_image = blurred_image[top_padding: top_padding+h, left_padding:left_padding+w]
    sorted_corners = sort_corners_clockwise(corners)
    sorted_corners = [x-50 for x in sorted_corners]
    warped_image = perspective_transform(image, sorted_corners)

    image=warped_image

    ######################################################################
    ######################################################################
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    ######################################################################

    return image
