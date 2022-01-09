import cv2
from .colour_segmentation import segment_lanes
from .midlane_fix import fix_midlane_gaps
from config import config
from .cleaning import get_outer_lane_inner_edge, extend_short_lane
from .data_extraction import data_extraction

def detect_lanes(frame):
    # Only bottom half of image needed
    resize_height = int((630 / 1080) * 240)
    frame_cropped = frame[resize_height:,:]

    # Colour segmentation on lanes
    mid_lane_edge, outer_lane_edge, outer_lane_edges_sep, outer_lane_points = segment_lanes(frame_cropped, config.min_area_resized)

    # Fix gaps in the middle lane
    mid_lane_no_gaps = fix_midlane_gaps(mid_lane_edge, config.max_distance_resized)

    # Fix outer lane, and get only the closest edge of the outer lane
    outer_lane_one_side, outer_lane_cnts_one_side, mid_lane_cnts, offset = get_outer_lane_inner_edge(outer_lane_edges_sep, mid_lane_no_gaps, outer_lane_points)

    # Fix both lanes by extending line to reach bottom of image if required
    extended_mid_lane, extended_outer_lane = extend_short_lane(mid_lane_no_gaps, mid_lane_cnts, outer_lane_cnts_one_side, outer_lane_one_side.copy())

    # With new frame data, get information for moving the car
    distance_error, curvature = data_extraction(mid_lane_edge, extended_mid_lane, extended_outer_lane, frame_cropped, offset)

    cv2.imshow("MID LANE EDGE", mid_lane_edge)
    cv2.imshow("OUTER LANE EDGE", outer_lane_edge)

    cv2.imshow("MID LANE GAP FIX", mid_lane_no_gaps)
    cv2.imshow("OUTER LANE ONE SIDE", outer_lane_one_side)

    cv2.imshow("EXTENDED MID LANE", extended_mid_lane)
    cv2.imshow("EXTENDED OUTER LANE", extended_outer_lane)
    
    cv2.waitKey(1)

    return distance_error, curvature
    


