import cv2
import numpy as np

def d(a, b):
    return np.sqrt(np.sum((a-b)**2))

def check_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use the Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,            # Inverse ratio of the accumulator resolution to the image resolution (1 means same resolution)
        minDist=50,      # Minimum distance between detected centers
        param1=150,      # Upper threshold for the internal Canny edge detector
        param2=40,       # Threshold for center detection (lower means more circles will be detected)
        minRadius=20,    # Minimum radius of the circles
        maxRadius=100    # Maximum radius of the circles
    )

    if circles is not None:
       return np.zeros_like(image)
    # print(circles)

    # cv2.imshow('img', image)
    # cv2.waitKey(0)

    if circles is None:
        return True
    else:
        return False

def solution(image_path):
    image = cv2.imread(image_path)
    # cv2.imshow("imag", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use the Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,            # Inverse ratio of the accumulator resolution to the image resolution (1 means same resolution)
        minDist=50,      # Minimum distance between detected centers
        param1=150,      # Upper threshold for the internal Canny edge detector
        param2=40,       # Threshold for center detection (lower means more circles will be detected)
        minRadius=20,    # Minimum radius of the circles
        maxRadius=100    # Maximum radius of the circles
    )

    if circles is not None:
        return np.zeros_like(image)
    image = cv2.GaussianBlur(image, (5,5), 0)
    image = cv2.medianBlur(image, 5)
    b,g,r = cv2.split(image)
    r_max = np.max(r)
    r_th = 13
    sim_th = 85
    boundary = set({})

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j,2] > r_max*(1- r_th/100.0) and image[i,j,1] < 200 and image[i,j,0] < 100:
                boundary.add((i,j))

    # boundary = initialize_seeds(image, r_max, r_th)
    # cv2.imshow("bound", construct_mask_from_region(image, boundary))
    region = boundary.copy()
    mean = np.array([0,0,0])
    for i,j in region:
        mean += image[i,j]
    mean = mean/len(region)


    
    
    # cv2.imshow("image", construct_mask_from_region(image, region))

    it = 0 
    max_iter = 50000

    while True:
        it+=1
        # cv2.imshow(f"boundary{it}", construct_mask_from_region(image, boundary))
        # new_region, boundary = find_new_region(image, region, boundary, mean, sim_th)

        # mean = find_mean(image, new_region)
        new_region = region.copy()
        b = boundary.copy()
    # new_boundary = set({})
        for p in b.copy():
            i, j = p
            dx = [1,0,-1,0,1,1,-1,-1]
            dy = [0,1,0,-1,1,-1,1,-1]
            for k in range(8):
                x = i+dx[k]
                y = j+dy[k]
                if x<0 or x>=image.shape[0] or y<0 or y>=image.shape[1]:
                    continue
                # print("DEBUG ", image[x,y,2])
                if d(image[x,y], mean) < sim_th:
                    new_region.add((x,y))
                    if (x,y) not in region:
                        b.add((x,y))
            b.remove((i,j))
        boundary = b

        mean = np.array([0,0,0])
        for i,j in region:
            mean += image[i,j]
        mean = mean/len(region)

        # cv2.imshow("mask", construct_mask_from_region(image, new_region))
        if (new_region == region or it>max_iter):
            # print("ITER ", it)
            break
        region = new_region
    
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i,j in region:
        mask[i,j] = 255
    # cv2.imshow("mask", mask)

    kernel = np.ones((7,7),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fill = mask.copy()

    h, w = fill.shape[:2]
    m = np.zeros((h+2, w+2), np.uint8)
    kernel = np.ones((7,7),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5)), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, (255), thickness=cv2.FILLED)
    # mask = mask | fill_inv  

    mask = filled
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("maskclose", mask)
    # cv2.waitKey(0)

    return mask


