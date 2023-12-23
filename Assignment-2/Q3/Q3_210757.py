import cv2
import numpy as np


def solution(image_path):
    image = cv2.imread(image_path)
    # image = preprocess(image)
    
    
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, k)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    image = cv2.filter2D(image, -1, sharp_kernel)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    obj = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)[1]
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)  


    object_mask = obj


    contours = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    num_contours = len(contours)
    if num_contours != 1:
        return 'fake'
    

    bounding_box = cv2.boundingRect(object_mask)
    image = image[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
    object_mask = object_mask[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
    # cv2.imshow('image', image)

    peaks=[]
    valleys=[]

    points = []

    # cv2.imshow("obj", obj)
    # cv2.waitKey(0)

    for c in range(obj.shape[1]):
        top = np.argmax(obj[:, c])
        points.append((c, top))

    filter = int(0.01*(obj.shape[0]))
    # print(obj)
    downsample = int(0.03*(obj.shape[0]))

    # points = points[filter:-filter:downsample]
    # print(points)
    points = points[3:-3 :4]

    p = []

    for i in range(1, len(points)):
        if points[i][1] != points[i-1][1]:
            p.append(points[i])

    points = p

    # print(points)

    valleys.append(points[0])
    for i in range(1, len(points)-1):
        if points[i][1] > points[i-1][1] and points[i][1] > points[i+1][1]:
            valleys.append(points[i])
        if points[i][1] < points[i-1][1] and points[i][1] < points[i+1][1]:
            peaks.append(points[i])
    
    
    valleys.append(points[-1])
  
    # cv2.imshow("obj", object_mask)
    # cv2.waitKey(0)
    # peaks, valleys = find_points(object_mask)

    # print(len(peaks))
    # check_faces(image, object_mask, peaks, valleys)

    if len(peaks) != 10:
        return 'fake'
    
    lowest_line = object_mask[object_mask.shape[0]-4, :]
    left = np.argmax(lowest_line)
    right = object_mask.shape[1] - np.argmax(np.flip(lowest_line))
    center = (left+right)//2

    center_peak = 0
    for i in range(len(peaks)):
        if np.abs(peaks[i][0]-center) < np.abs(peaks[center_peak][0]-center):
            center_peak = i

    if center_peak !=4:
        return 'fake'

    # if not head_check(object_mask, peaks, valleys):
    #     print("Head count")
    #     return 'fake'


    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([17, 10, 0])
    upper_yellow = np.array([36, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    

    crown_contour = np.zeros_like(image)
    crown_contour = cv2.drawContours(crown_contour, [largest_contour], -1, (255,255,255), thickness=cv2.FILLED)
    k_size = int(0.035*object_mask.shape[0])
    kernel = np.ones((k_size, k_size), np.uint8)  # You may need to adjust the kernel size
    crown_contour = cv2.morphologyEx(crown_contour, cv2.MORPH_OPEN, kernel)
    crown_contour = cv2.morphologyEx(crown_contour, cv2.MORPH_CLOSE, kernel)
    
    crown_contour = cv2.cvtColor(crown_contour, cv2.COLOR_BGR2GRAY)

    face_mask= np.zeros_like(object_mask)

    for c in range(crown_contour.shape[1]):
        r = np.flip(crown_contour[:, c])
        top = np.argmax(r)
        face_mask[crown_contour.shape[0]-top:, c-1] = 255

    chins = []
    for c in range(object_mask.shape[1]):
        r = np.flip(object_mask[:, c])
        top = np.argmax(r)
        chins.append((c, object_mask.shape[0]-top))

    l = len(chins)
    trim = 0.03
    chins = chins[int(trim*l):int((1-trim)*l)]
    flanks = 0.2
    bias = 0
    sum=0
    sum +=np.sum([y for (x,y) in chins[:int(flanks*l)]])
    sum += np.sum([y for (x,y) in chins[-int(flanks*l):]])
    avg = sum/(2*flanks*l)
    avg = int(avg)
    avg+=bias


    chin_line = avg
    bottom_mask = np.zeros_like(object_mask)

    bottom_mask[0:chin_line, :] = 255
    # cv2.imshow("bottom", bottom_mask)
    # return bottom_mask

    
    

    mask = object_mask & face_mask & bottom_mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=2)
    # cv2.imshow("mask", mask)

    # masked_image = cv2.bitwise_and(image, image, mask=mask)


    image = cv2.filter2D(image, -1, sharp_kernel)

    for f in range(len(valleys)-1):
        x1 = valleys[f][0]
        x2 = valleys[f+1][0]
        # print(x1, x2)
        face_mask = np.zeros_like(mask)

        h,w = mask.shape
        cv2.rectangle(face_mask, (x1, 0), (x2, h), (255), thickness=cv2.FILLED)
        # mask = np.bitwised_and(mask, obj)

        # mask = np.bitwise_and(mask, )

        # mask = np.bitwise_and(mask, )

        face_mask = face_mask & mask
        mask_area = np.sum(mask)/255
        k_size = int(0.00025*mask_area)

        face_mask = cv2.erode(face_mask, np.ones((3,3),np.uint8), iterations=2)

        face = cv2.bitwise_and(image, image, mask=face_mask)
        # face = cv2.GaussianBlur(face, (5, 5), 0)
        # face = cv2.medianBlur(face, 5)
        sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        face = cv2.filter2D(face, -1, sharp_kernel)
        # face = sharpen(face)
        face = cv2.GaussianBlur(face, (3, 3), 0)
        face = cv2.GaussianBlur(face, (3, 3), 0)
        canny = cv2.Canny(face, 160, 200)

        contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        no_of_contours = len(contours)
        face_area = np.sum(face_mask)
        canny_area = np.sum(canny)**2

        center_adjustment = 10*np.log(1+(np.abs(f-4)))
        face_index = canny_area/(face_area*100)
        face_index +=center_adjustment  

        # cv2.imshow(f"canny{f} - {face_index}", canny)
        # cv2.imshow(f"face_mask{f}", face)
        # cv2.waitKey(0)
        # contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # continue
        if face_index < 100:
            return 'fake'
    
    # cv2.imshow("masked", masked_image)
    # cv2.waitKey(0)



    
    return 'real'


    



   