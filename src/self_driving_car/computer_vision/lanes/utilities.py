import numpy as np
import cv2
import math

def get_distance(a, b):
    return math.sqrt( ((a[1] - b[1])**2) + ((a[0] - b[0])**2) )

def get_lane_curvature(low_x, low_y, high_x, high_y):
    # Default values
    slope=1000
    y_intercept = 0
    angle_of_inclination = 90
    curvature = 0

    if((high_x - low_x) != 0): # If trajectory exists
        slope = (high_y - low_y) / (high_x - low_x) # get the gradient
        y_intercept = high_y - (slope * high_x) # use y = mx + c
        angle_of_inclination = math.atan(slope) * (180 / np.pi) # Convert to degrees

    if(angle_of_inclination != 90): # If angle_of_inclination was updated
        if(angle_of_inclination < 0): # curvature is angled right
            curvature = 90 + angle_of_inclination
        else: # curvature is angled left
            curvature = angle_of_inclination - 90

    return curvature

def remove_image_noise(img, min_area):
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts_too_small = []
    for index, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area < min_area:
            cnts_too_small.append(cnt)
    
    thresh = cv2.drawContours(thresh, cnts_too_small, -1, 0, -1)
    return thresh

def find_extremes(img):
    positions = np.nonzero(img)
    if (len(positions) != 0):
        top = positions[0].min()
        bottom = positions[0].max()
        left = positions[1].min()
        right = positions[1].max()
        return top, bottom
    else:
        return 0, 0

def find_lowest_row(img):
    positions = np.nonzero(img)
    
    if (len(positions) != 0):
        top = positions[0].min()
        bottom = positions[0].max()
        left = positions[1].min()
        right = positions[1].max()
        return bottom
    else:
        return img.shape[0]

def get_largest_contour_of_outer_lane(outer_lane, min_area):
    largest_contour_found = False
    thresh = np.zeros(outer_lane.shape, dtype=outer_lane.dtype)
    _, bin_img = cv2.threshold(outer_lane, 0, 255, cv2.THRESH_BINARY)
    
    # Dilating images to prevent shadows causing problems
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5,5))
    bin_img_dilated = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel)
    bin_img_ret = cv2.morphologyEx(bin_img_dilated, cv2.MORPH_ERODE, kernel)
    bin_img = bin_img_ret

    cnts = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    max_cnt_area = 0
    max_cnt_idx = -1
    for index, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > max_cnt_area:
            max_cnt_area = area
            max_cnt_idx = index
            largest_contour_found = True
    
    # Update mask if largest contour
    if max_cnt_area < min_area:
        largest_contour_found = False
    if ((max_cnt_idx != -1) and (largest_contour_found)):
        thresh = cv2.drawContours(thresh, cnts, max_cnt_idx, (255,255,255), -1)
    return thresh, largest_contour_found

def extract_roi(image, start_point, end_point):
    roi = np.zeros(image.shape, dtype=np.uint8)
    cv2.rectangle(roi, start_point, end_point, 255, thickness = -1)
    final_roi = cv2.bitwise_and(image, roi)
    return final_roi

def extract_point(img, specified_row):
    Point= (0,specified_row)
    specified_row_data = img[specified_row-1,:]
    positions = np.nonzero(specified_row_data)  
    if (len(positions[0]) != 0):
        min_col = positions[0].min()
        Point = (min_col, specified_row)
    return Point

def get_lowest_edge_points(outer_lane_):
    outer_points_list=[]
    thresh = np.zeros(outer_lane_.shape, dtype=outer_lane_.dtype)
    lane_side_one = np.zeros(outer_lane_.shape, dtype=outer_lane_.dtype)
    lane_side_two = np.zeros(outer_lane_.shape, dtype=outer_lane_.dtype)
    _, bin_img = cv2.threshold(outer_lane_, 0, 255, cv2.THRESH_BINARY)

    # Find the two contours for which you want to find the min distance between them
    cnts = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    thresh = cv2.drawContours(thresh, cnts, 0, (255,255,255), 1)

    # Boundary of the Contour is extracted and saved in thresh
    top_row, bot_row = find_extremes(thresh)

    roi = extract_roi(thresh, (0, top_row + 5), (thresh.shape[1], bot_row-5))
    cnts2 = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    low_row_a = -1
    low_row_b = -1
    euc_row = 0

    first_line = np.copy(lane_side_one)
    cnts_tmp = []

    if(len(cnts2) > 1):
        for index_tmp, cnt_tmp in enumerate(cnts2):
            if((cnt_tmp.shape[0]) > 50):
                cnts_tmp.append(cnt_tmp)
        cnts2 = cnts_tmp

    for index, cnt in enumerate(cnts2):
        lane_side_one = np.zeros(outer_lane_.shape, dtype=outer_lane_.dtype)
        lane_side_one = cv2.drawContours(lane_side_one, cnts2, index, (255,255,255), 1) 
        lane_side_two = cv2.drawContours(lane_side_two, cnts2, index, (255,255,255), 1)

        if(len(cnts2) == 2):
            if (index == 0):
                first_line = np.copy(lane_side_one)
                low_row_a = find_lowest_row(lane_side_one)
            elif(index == 1):
                low_row_b = find_lowest_row(lane_side_one)
                if(low_row_a < low_row_b):
                    euc_row=low_row_a
                else:
                    euc_row=low_row_b
                point_a = extract_point(first_line, euc_row)
                point_b = extract_point(lane_side_one, euc_row)
                outer_points_list.append(point_a)
                outer_points_list.append(point_b)
    
    return lane_side_two, outer_points_list

def findLineParameter(x1,y1,x2,y2):
    if((x2-x1)!=0):
        slope = (y2-y1)/(x2-x1)
        y_intercept = y2 - (slope*x2) #y= mx+c
    else:
        slope=1000
        y_intercept=0
        #print("Vertical Line [Undefined slope]")
    return (slope,y_intercept)

def update_contours(cnts, order):

    if cnts:
        cnt = cnts[0]
        cnt = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
        order_list = []
        if(order == "rows"):
            order_list.append((0, 1))
        else:
            order_list.append((1, 0))
        ind = np.lexsort((cnt[:,order_list[0][0]], cnt[:,order_list[0][1]]))
        Sorted = cnt[ind]
        return Sorted
    else:
        return cnts
