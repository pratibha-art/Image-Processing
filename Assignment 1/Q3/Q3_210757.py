import cv2
import numpy as np
def add_padding(image, top, bottom, left, right):
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded_image
def solution(image_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,w= image.shape
    top_padding = 50
    bottom_padding = 50
    left_padding = 50
    right_padding = 50
    padded_image = add_padding(image, top_padding, bottom_padding, left_padding, right_padding)
    _, thresh = cv2.threshold(padded_image, 250, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((17,17), np.uint8)
    d = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    combined_contour = np.concatenate(contours)
    hull = cv2.convexHull(combined_contour)
    rect = cv2.minAreaRect(hull)
    output_image = padded_image.copy()
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(output_image, [box], 0, (0 ,255, 0), 2)
    angle = rect[2]
    _, thres = cv2.threshold(output_image, 250, 255, cv2.THRESH_BINARY_INV)
    angle = rect[2]
    height, width = rect[1]
    offset = 0 if height> width else 90
    image_center = tuple(np.array(thresh.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, offset + angle, 1.0)
    result = cv2.warpAffine(thresh, rot_mat, thres.shape[1::-1], flags=cv2.INTER_LINEAR)
    inverted_threshold = cv2.bitwise_not(result)
    image=inverted_threshold
    return image
