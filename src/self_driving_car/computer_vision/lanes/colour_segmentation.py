import cv2
import numpy as np
from .utilities import remove_image_noise, get_lowest_edge_points, get_largest_contour_of_outer_lane

# Globals
frame_hsl = 0

# HSL White Regions Range (middle of lane)
HUE_LOW_WHITE = 0
LIT_LOW_WHITE = 225
SAT_LOW_WHITE = 0

# HSL Yellow Regions Range (side of lane)
HUE_LOW_YELLOW = 30
HUE_HIGH_YELLOW = 35
LIT_LOW_YELLOW = 150
SAT_LOW_YELLOW = 0


def colour_segmentation(frame_hsl, lower_range, upper_range):
    # Segment based on colour
    mask_in_range = cv2.inRange(frame_hsl, lower_range, upper_range)

    # Perform morphology (dilation) to help with noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    final_mask = cv2.morphologyEx(mask_in_range, cv2.MORPH_DILATE, kernel)
    return final_mask


def get_updated_frame_and_edge(full_frame, region_frame, min_area):
    # Combine frames and greyscale image
    frame_updated = cv2.bitwise_and(full_frame, full_frame, mask=region_frame)
    frame_updated_gray = cv2.cvtColor(frame_updated, cv2.COLOR_BGR2GRAY)

    # Remove noise
    refined_frame = remove_image_noise(frame_updated_gray, min_area)

    # Get edges using Canny Edge Detection, but first removing noiseusing Gaussian filter
    frame_updated_gray = cv2.bitwise_and(frame_updated_gray, refined_frame)
    frame_smoothed = cv2.GaussianBlur(frame_updated_gray,(11,11), 1)
    edges_of_refined_frame = cv2.Canny(frame_smoothed, 50, 150, None, 3)

    return refined_frame, edges_of_refined_frame


def refine_mid_lane(full_frame, white_region_frame, min_area):
    _, mid_lane_edge = get_updated_frame_and_edge(full_frame, white_region_frame, min_area)
    return mid_lane_edge


def refine_outer_lane(full_frame, yellow_region_frame, min_area):
    outer_points_list = []
    outer_lane_mask, outer_lane_edges = get_updated_frame_and_edge(full_frame, yellow_region_frame, min_area)

    # Ensure correct outer lane is being used (to the RIGHT of the car, the largest one)
    outer_lane_mask_largest, largest_found = get_largest_contour_of_outer_lane(outer_lane_mask, min_area)

    if largest_found:
        # Keep only edges of largest outer lane 
        outer_lane_edge_largest = cv2.bitwise_and(outer_lane_edges, outer_lane_mask_largest)
        outer_lane_edges_sep, outer_points_list = get_lowest_edge_points(outer_lane_mask_largest)
        outer_lane_edges = outer_lane_edge_largest
    else:
        outer_lane_edges_sep = np.zeros((full_frame.shape[0], full_frame.shape[1]), np.uint8)
    
    return outer_lane_edges, outer_lane_edges_sep, outer_points_list


def segment_lanes(full_frame, min_area):
    global frame_hsl

    # Convert colour space of frame captured by Gazebo camera from RGB to HLS
    frame_hsl = cv2.cvtColor(full_frame, cv2.COLOR_BGR2HLS)

    # Segment each side of the road
    white_region_frame = colour_segmentation(frame_hsl, np.array([HUE_LOW_WHITE, LIT_LOW_WHITE, SAT_LOW_WHITE]), np.array([255, 255, 255]))
    yellow_region_frame = colour_segmentation(frame_hsl, np.array([HUE_LOW_YELLOW, LIT_LOW_YELLOW, SAT_LOW_YELLOW]), np.array([HUE_HIGH_YELLOW, 255, 255]))

    # Display Regions for debugging
    cv2.imshow("MID LANE, WHITE REGION", white_region_frame)
    cv2.imshow("OUTER LANE, YELLOW REGION", yellow_region_frame)
    cv2.waitKey(1)

    # Remove noise from each segment and get the edge of the regions
    mid_lane_edge = refine_mid_lane(full_frame, white_region_frame, min_area)
    outer_lane_edge, outer_lane_edges_sep, outer_lane_points = refine_outer_lane(full_frame, yellow_region_frame, min_area + 500)        

    return mid_lane_edge, outer_lane_edge, outer_lane_edges_sep, outer_lane_points


############################## ADD TRACKBAR TO GET COLOUR RANGE REQUIRED ##############################

def update_window():
    white_region = colour_segmentation(frame_hsl, (HUE_LOW_WHITE, LIT_LOW_WHITE, SAT_LOW_WHITE), (255, 255, 255))
    yellow_region = colour_segmentation(frame_hsl, (HUE_LOW_YELLOW, LIT_LOW_YELLOW, SAT_LOW_YELLOW), (HUE_HIGH_YELLOW, 255, 255))
    cv2.imshow("MID LANE, WHITE REGION", white_region)
    cv2.imshow("OUTER LANE, YELLOW REGION", yellow_region)


def on_hue_low_change(val):
    global HUE_LOW_WHITE
    HUE_LOW_WHITE = val
    update_window()
def on_lit_low_change(val):
    global LIT_LOW_WHITE
    LIT_LOW_WHITE = val
    update_window()
def on_sat_low_change(val):
    global SAT_LOW_WHITE
    SAT_LOW_WHITE = val
    update_window()
    
def on_hue_low_y_change(val):
    global HUE_LOW_YELLOW
    HUE_LOW_YELLOW = val
    update_window()
def on_hue_high_y_change(val):
    global HUE_HIGH_YELLOW
    HUE_HIGH_YELLOW = val
    update_window()
def on_lit_low_y_change(val):
    global LIT_LOW_YELLOW
    LIT_LOW_YELLOW = val
    update_window()
def on_sat_low_y_change(val):
    global SAT_LOW_YELLOW
    SAT_LOW_YELLOW = val
    update_window()

cv2.namedWindow("MID LANE, WHITE REGION")
cv2.namedWindow("OUTER LANE, YELLOW REGION")

cv2.createTrackbar("HUE_LOW_WHITE", "MID LANE, WHITE REGION", HUE_LOW_WHITE, 255, on_hue_low_change)
cv2.createTrackbar("LIT_LOW_WHITE", "MID LANE, WHITE REGION", LIT_LOW_WHITE, 255, on_lit_low_change)
cv2.createTrackbar("SAT_LOW_WHITE", "MID LANE, WHITE REGION", SAT_LOW_WHITE, 255, on_sat_low_change)

cv2.createTrackbar("HUE_LOW_YELLOW", "OUTER LANE, YELLOW REGION", HUE_LOW_YELLOW, 255, on_hue_low_y_change)
cv2.createTrackbar("HUE_HIGH_YELLOW", "OUTER LANE, YELLOW REGION", HUE_HIGH_YELLOW, 255, on_hue_high_y_change)
cv2.createTrackbar("LIT_LOW_YELLOW", "OUTER LANE, YELLOW REGION", LIT_LOW_YELLOW, 255, on_lit_low_y_change)
cv2.createTrackbar("SAT_LOW_YELLOW", "OUTER LANE, YELLOW REGION", SAT_LOW_YELLOW, 255, on_sat_low_y_change)
