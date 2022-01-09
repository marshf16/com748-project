import cv2
import numpy as np
from .utilities import get_distance, update_contours

def is_car_trajectory_crossing_midlane(mid_lane, mid_lane_cnts, outer_lane_cnts):
    is_crossing_left = 0
    car_path_image_reference = np.zeros_like(mid_lane)
    mid_lane_copy = mid_lane.copy()
    
    # Get the bottom point (Y axis = 0) for each side of the lane in the frame
    mid_lane_cnts_rows_sorted = update_contours(mid_lane_cnts, "rows")
    outer_lane_cnts_rows_sorted = update_contours(outer_lane_cnts, "rows")
    mid_rows = mid_lane_cnts_rows_sorted.shape[0]
    outer_rows = outer_lane_cnts_rows_sorted.shape[0]
    mid_lane_bottom_point = mid_lane_cnts_rows_sorted[mid_rows-1,:]
    outer_lane_bottom_point = outer_lane_cnts_rows_sorted[outer_rows-1,:]

    # Get car trajectory by using the bottom points found
    car_trajectory = (int((mid_lane_bottom_point[0] + outer_lane_bottom_point[0]) / 2), int((mid_lane_bottom_point[1] + outer_lane_bottom_point[1]) / 2 ))

    # Draw line between car position to car trajectory
    cv2.line(car_path_image_reference, car_trajectory, (int(car_path_image_reference.shape[1] / 2), car_path_image_reference.shape[0]), (255,255,0), 2)

    # Draw line between mid lane to bottom
    cv2.line(mid_lane_copy, tuple(mid_lane_bottom_point), (mid_lane_bottom_point[0], mid_lane_copy.shape[0] - 1), (255,255,0), 2)
    
    # Check if car trajectory is crossing mid lane
    is_crossing_left = ( (int(car_path_image_reference.shape[1] / 2) - car_trajectory[0]) > 0 )

    if(np.any((cv2.bitwise_and(car_path_image_reference, mid_lane_copy) > 0))): # If true, the mid lane and the car path intersects
        return True, is_crossing_left
    else:
        return False, is_crossing_left

def extend_short_lane(mid_lane, mid_lane_cnts, outer_lane_cnts, outer_lane):
    if(mid_lane_cnts and outer_lane_cnts):
        # Sort both lanes contours by rows and count them
        mid_lane_cnts_rows_sorted = update_contours(mid_lane_cnts, "rows")
        outer_lane_cnts_rows_sorted = update_contours(outer_lane_cnts, "rows")
        bottom_of_image = mid_lane.shape[0]
        total_cnts_midlane = mid_lane_cnts_rows_sorted.shape[0]
        total_cnts_outerlane = outer_lane_cnts_rows_sorted.shape[0]

        # If not connected, draw line between mid lane and bottom of image
        lowest_midlane_point = mid_lane_cnts_rows_sorted[total_cnts_midlane-1,:]
        if (lowest_midlane_point[1] < bottom_of_image):
            mid_lane = cv2.line(mid_lane, tuple(lowest_midlane_point), (lowest_midlane_point[0], bottom_of_image), 255, 2)

        # If not connected, draw line between outer lane and bottom of image
        lowest_outerlane_point = outer_lane_cnts_rows_sorted[total_cnts_outerlane-1,:]
        if (lowest_outerlane_point[1] < bottom_of_image):
            if(total_cnts_outerlane > 20): # Taking last 20 points for estimation
                shift=20
            else:
                shift=2

            ref_last_points = outer_lane_cnts_rows_sorted[total_cnts_outerlane-shift:total_cnts_outerlane-1:2,:]

            # Estimate slope
            if(len(ref_last_points) > 1): # At least 2 points needed to estimate a line
                ref_x = ref_last_points[:,0] # cols
                ref_y = ref_last_points[:,1] # rows
                ref_params = np.polyfit(ref_x, ref_y, 1)
                ref_slope = ref_params[0]
                ref_y_intercept = ref_params[1]

                # Extend outerlane in the direction of its slope
                if(ref_slope < 0):
                    ref_line_touching_point_col = 0
                    ref_line_touching_point_row = ref_y_intercept
                else:
                    ref_line_touching_point_col = outer_lane.shape[1] - 1 # Cols have length of ColLength But traversal is from 0 to ColLength-1
                    ref_line_touching_point_row = ref_slope * ref_line_touching_point_col + ref_y_intercept

                # Finally draw line between outer lane and bottom of image
                ref_touch_point = (ref_line_touching_point_col, int(ref_line_touching_point_row))
                ref_bottom_point_tuple = tuple(lowest_outerlane_point)
                outer_lane = cv2.line(outer_lane, ref_touch_point, ref_bottom_point_tuple, 255, 2)

                # If required, connect outerlane to bottom by drawing a vertical line
                if(ref_line_touching_point_row < bottom_of_image):
                    ref_touch_point_ref = (ref_line_touching_point_col, bottom_of_image)
                    outer_lane = cv2.line(outer_lane, ref_touch_point, ref_touch_point_ref, 255, 3)

    return mid_lane, outer_lane

def get_outer_lane_inner_edge(outer_lane, mid_lane, outer_lane_points):
    offset = 0

    # Declaration of frame which will contain the edge of the outer lane to use
    outer_lane_edge = np.zeros_like(outer_lane)

    # Get contours of mid and outer lane
    mid_lane_cnts = cv2.findContours(mid_lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    outer_lane_cnts = cv2.findContours(outer_lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    # Set boolean whether outer lane is currently present
    if not outer_lane_cnts:
        outer_lane_present = True
    else:
        outer_lane_present = False

    # Set first contour from midlane as reference
    if(mid_lane_cnts): # If mid lane contours are present, use the first contour point
        mid_lane_ref_point = tuple(mid_lane_cnts[0][0][0])
    else:
        mid_lane_ref_point = (0,0) 
    
    # Ensure midlane exists
    if mid_lane_cnts:
        # Condition 1: Both mid lane and outer lane is detected
        if  (len(outer_lane_points) == 2): # If both edges of outer lane detected, attempt removal of a line
            # Keep only the line on the innder side of the outer lane
            edge_a = outer_lane_points[0]
            edge_b = outer_lane_points[1]
            closest_idx = 0
            if(get_distance(edge_a, mid_lane_ref_point) <= get_distance(edge_b, mid_lane_ref_point)):
                closest_idx = 0
            elif(len(outer_lane_cnts) > 1):
                closest_idx = 1
            outer_lane_edge = cv2.drawContours(outer_lane_edge, outer_lane_cnts, closest_idx, 255, 1)
            outer_lane_cnts_updated = [outer_lane_cnts[closest_idx]]

            # Check if the correct outer lane was detected
            is_path_crossing , is_crossing_left = is_car_trajectory_crossing_midlane(mid_lane, mid_lane_cnts, outer_lane_cnts_updated)
            if(is_path_crossing):
                outer_lane = np.zeros_like(outer_lane) # wrong outer lane, turn into empty image
            else:
                return outer_lane_edge, outer_lane_cnts_updated, mid_lane_cnts, 0

        elif(np.any(outer_lane > 0) ): # If only 1 edge of outer lane detected, no need to attempt removal of a line
            # Check if the correct outer lane was detected
            is_path_crossing , is_crossing_left = is_car_trajectory_crossing_midlane(mid_lane, mid_lane_cnts, outer_lane_cnts)
            if(is_path_crossing):
                outer_lane = np.zeros_like(outer_lane) # wrong outer lane, turn into empty image
            else:
                return outer_lane, outer_lane_cnts, mid_lane_cnts, 0

        # Condition 2: Only mid lane is present, no outer lane
        if(not np.any(outer_lane > 0)):
            # Get bottom point of midlane
            mid_lane_cnts_rows_sorted = update_contours(mid_lane_cnts,"rows")
            mid_lane_rows = mid_lane_cnts_rows_sorted.shape[0]
            mid_lane_low_point = mid_lane_cnts_rows_sorted[mid_lane_rows-1,:]
            mid_lane_high_point = mid_lane_cnts_rows_sorted[0,:]
            mid_lane_bottom = mid_lane_low_point[0]

            # Decide which side to draw outer lane	
            draw_outer_lane_right = False
            if outer_lane_present:
                if(mid_lane_bottom < int(mid_lane.shape[1]/2)):
                    draw_outer_lane_right = True
            else:
                if is_crossing_left:
                    draw_outer_lane_right = True

            # Set upper and lower points of outer lane
            if draw_outer_lane_right:
                low_point = (int(mid_lane.shape[1])-1)
                high_point = (int(mid_lane.shape[1])-1)
                offset = 20
            else:
                low_point = 0
                high_point = 0
                offset = -20

            # Set outer lane points from which to draw line
            mid_lane_low_point[1] = mid_lane.shape[0]
            lane_lower_point = (low_point, int(mid_lane_low_point[1]))
            lane_upper_point = (high_point, int(mid_lane_high_point[1]))

            # Draw line for outer lane
            outer_lane = cv2.line(outer_lane, lane_lower_point, lane_upper_point, 255, 1)
            outer_lane_cnts = cv2.findContours(outer_lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            return outer_lane, outer_lane_cnts, mid_lane_cnts, offset

    # Ignore if no mid lane found
    else:
        return outer_lane, outer_lane_cnts, mid_lane_cnts, offset
   