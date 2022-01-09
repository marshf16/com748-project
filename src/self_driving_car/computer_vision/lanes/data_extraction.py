import cv2
import numpy as np
from .utilities import update_contours, get_lane_curvature

def get_road_trajectory(mid_lane, outer_lane, offset):
    mid_lane_cnts = cv2.findContours(mid_lane, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    outer_lane_cnts = cv2.findContours(outer_lane, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]

    if mid_lane_cnts and outer_lane_cnts:
        # Sort contours into rows
        mid_cnts_sorted = update_contours(mid_lane_cnts, "rows")
        outer_cnts_sorted = update_contours(outer_lane_cnts, "rows")
        mid_lane_rows = mid_cnts_sorted.shape[0] # number of rows/contours in mid lane
        outer_lane_rows = outer_cnts_sorted.shape[0] # number of rows/contours in outer lane

        # Get midlane and outerlane upper/lower points
        midlane_bottom_point = mid_cnts_sorted[mid_lane_rows - 1,:]
        outerlane_bottom_point = outer_cnts_sorted[outer_lane_rows - 1,:]
        midlane_top_point = mid_cnts_sorted[0,:]
        outerlane_top_point = outer_cnts_sorted[0,:]

        # Create trajectory
        trajectory_low_point = (int((midlane_bottom_point[0] + outerlane_bottom_point[0]) / 2) + offset, int((outerlane_bottom_point[1] + outerlane_bottom_point[1]) / 2))
        trajectory_high_point = (int((midlane_top_point[0] + outerlane_top_point[0]) / 2) + offset, int((outerlane_top_point[1] + outerlane_top_point[1]) / 2))

        return trajectory_low_point, trajectory_high_point

    else:
        return (0,0),(0,0)

def remove_mid_lane(mid_lane_edge):
    no_mid_lane = np.zeros((mid_lane_edge.shape[0], mid_lane_edge.shape[1], 1), dtype=np.uint8)
    cnts = cv2.findContours(mid_lane_edge, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
    if cnts:
        hull_list = []
        cnts = np.concatenate(cnts)
        hull = cv2.convexHull(cnts)
        hull_list.append(hull)
        no_mid_lane = cv2.drawContours(no_mid_lane, hull_list, 0, 255,-1)
    lane_without_midlane = cv2.bitwise_not(no_mid_lane)
    return lane_without_midlane

def data_extraction(mid_lane_edge, mid_lane, outer_lane, frame, offset):
    # Create path estimation (road trajectory) using both sides of the lane
    trajectory_low_point, trajectory_high_point = get_road_trajectory(mid_lane, outer_lane, offset)
    
    # Get distance error value (distance between start car location and start road trajectory location)
    # and curvature value (angle between a vertical line and the road trajectory)
    distance_error = -1000
    if(trajectory_low_point != (0,0)):
        distance_error = trajectory_low_point[0] - int(mid_lane.shape[1] / 2)
    curvature = get_lane_curvature(trajectory_low_point[0], trajectory_low_point[1], trajectory_high_point[0], trajectory_high_point[1])

    # Ensure only edges apart of the mid lane
    mid_lane_edge = cv2.bitwise_and(mid_lane_edge, mid_lane)

    # Combine both sides of the lane for viewing
    lanes_combined = cv2.bitwise_or(outer_lane, mid_lane)
    cv2.imshow("LANES COMBINED", lanes_combined)

    # Get contours of combiend lanes to draw projected driving area
    projected_lane = np.zeros(lanes_combined.shape, lanes_combined.dtype)
    cnts = cv2.findContours(lanes_combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

    # Fill projected driving area
    if cnts:
        cnts = np.concatenate(cnts)
        cnts = np.array(cnts)
        cv2.fillConvexPoly(projected_lane, cnts, 255)

    # Remove mid_lane mask from projected driving area (the white road paint should not be inside projected area)
    lane_without_midlane = remove_mid_lane(mid_lane_edge)
    projected_lane = cv2.bitwise_and(lane_without_midlane, projected_lane)

    # Draw projected driving area in green
    projected_driving_frame = frame
    projected_driving_frame[projected_lane==255] = projected_driving_frame[projected_lane==255] + (0,100,0)

    # Draw car position line
    cv2.line(projected_driving_frame, 
            (int(projected_driving_frame.shape[1] / 2), projected_driving_frame.shape[0]), 
            (int(projected_driving_frame.shape[1] / 2), projected_driving_frame.shape[0] - int (projected_driving_frame.shape[0] / 5)), 
            (0,0,255), 
            2)

    # Draw road trajectory line
    cv2.line(projected_driving_frame, trajectory_low_point, trajectory_high_point, (255,0,0),2)

    # Show distance and curvature on window for debugging purposes
    # curvature_text = "Curvature = " + "{:.2f}".format(curvature)
    # distance_error_text = "Distance = " + str(distance_error)
    # cv2.putText(projected_driving_frame, curvature_text, (10,30), cv2.FONT_ITALIC, 0.5, (0,0,255), 1)
    # cv2.putText(projected_driving_frame, distance_error_text, (10,50), cv2.FONT_ITALIC, 0.5, (0,0,255), 1)

    return distance_error, curvature

