import cv2
import math
import numpy as np
from .utilities import get_distance

def get_largest_contour(frame_grey):
    largest_contour_found = False
    thresh = np.zeros(frame_grey.shape, dtype=frame_grey.dtype)
    _, bin_img = cv2.threshold(frame_grey, 0, 255, cv2.THRESH_BINARY)

    #Find the two contours where you want to find the minimum distance
    cnts = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    max_cnt_area = 0
    max_cnt_idx = -1
    for index, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > max_cnt_area:
            max_cnt_area = area
            max_cnt_idx = index
            largest_contour_found = True

    if (max_cnt_idx != -1):
        thresh = cv2.drawContours(thresh, cnts, max_cnt_idx, (255,255,255), -1)
    return thresh, largest_contour_found

def get_distance_between_patches(cnt1, cnt2):
    # Get the centroid of each patch using moments from OpenCV
    M_cnt1 = cv2.moments(cnt1)
    cX_1 = int(M_cnt1["m10"] / M_cnt1["m00"])
    cY_1 = int(M_cnt1["m01"] / M_cnt1["m00"])
    
    M_cnt2 = cv2.moments(cnt2)
    cX_2 = int(M_cnt2["m10"] / M_cnt2["m00"])
    cY_2 = int(M_cnt2["m01"] / M_cnt2["m00"])

    # Get minimum distance between the 2 centre points
    centre_point_a = (cX_1, cY_1)
    centre_point_b = (cX_2, cY_2)
    minimum_distance = get_distance(centre_point_a, centre_point_b)
    return minimum_distance, centre_point_a, centre_point_b

def fix_midlane_gaps(midlane_patches, max_dist):
    # Dilate image to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    midlane_patches = cv2.morphologyEx(midlane_patches, cv2.MORPH_DILATE, kernel)

    # Get BGR image to draw connecting patches
    midlane_connectivity_bgr = cv2.cvtColor(midlane_patches, cv2.COLOR_GRAY2BGR)

    # Get contours from the patches
    cnts = cv2.findContours(midlane_patches, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

    # Remove any foriegn objects in image using minimum area, keeping only contours of the mid lane patches
    min_area = 1
    patch_cnts_list = []
    for _, cnt in enumerate(cnts):
        cnt_area = cv2.contourArea(cnt)
        if (cnt_area > min_area):
            patch_cnts_list.append(cnt)
    cnts = patch_cnts_list

    ##### Connect each patch to the next closest patch #####
    cnt_idx_best_match = []

    # Start looping through contour list
    for index, cnt_original in enumerate(cnts):
        previous_minimum_dist = 100000
        best_idx_compared = 0
        best_centre_point_a = 0
        best_centre_point_b = 0

        # Start looping through other contours to find closest
        for index_compared in range(len(cnts) - index):
            index_compared = index_compared + index
            cnt_compared = cnts[index_compared]

            if (index != index_compared): # Ensure not the same contour being compared
                min_dist, centre_cnt1, centre_cnt2 = get_distance_between_patches(cnt_original, cnt_compared)
                if (min_dist < previous_minimum_dist): # If true, closer contour has been found
                    if (len(cnt_idx_best_match) == 0): # If true, no 'closest contour' found yet
                        previous_minimum_dist = min_dist
                        best_idx_compared = index_compared
                        best_centre_point_a = centre_cnt1
                        best_centre_point_b = centre_cnt2
                    else: # If other 'closest contour' has been found
                        connection_present = False
                        for i in range(len(cnt_idx_best_match)):
                            if ( (index_compared == i) and (index == cnt_idx_best_match[i]) ): # If true, ignore this match as connection already present
                                connection_present = True
                        if not connection_present: # No connection, update values
                            previous_minimum_dist = min_dist
                            best_idx_compared = index_compared
                            best_centre_point_a = centre_cnt1
                            best_centre_point_b = centre_cnt2

        if ((previous_minimum_dist != 100000) and (previous_minimum_dist > max_dist)): # If previous minimum distance is greater than the maximum allowed distance, break loop
            break
        if (type(best_centre_point_a) != int): #If true, centroid point was updated as better contour match was found
            cnt_idx_best_match.append(best_idx_compared)

            # Finally, connect patches/contours with a white line of thickness 2
            cv2.line(midlane_connectivity_bgr, best_centre_point_a, best_centre_point_b,( 0,255,0), 2)

    # Convert new frame with drawn line back into greyscale
    midlane_connectivity = cv2.cvtColor(midlane_connectivity_bgr, cv2.COLOR_BGR2GRAY)

    # Get estimated midlane by returning the largest contour
    estimated_midlane, largest_found = get_largest_contour(midlane_connectivity)

    if largest_found: # Return new frame with new data
        return estimated_midlane
    else: # Otherwise return original frame
        return midlane_patches