import os
import cv2

# Configuration values for Lane Detection
resized_width = 320
resized_height = 240
ref_frame_width = 1920
ref_frame_height = 1080
ref_frame_dimension = ref_frame_width * ref_frame_height
resized_frame_dimension = resized_width * resized_height
lane_extraction_min_area_per = 1000 / ref_frame_dimension
min_area_resized = int(resized_frame_dimension * lane_extraction_min_area_per)

# Configuration values for gap fixing
max_dist_per = 600 / ref_frame_height
max_distance_resized = int(resized_height * max_dist_per)