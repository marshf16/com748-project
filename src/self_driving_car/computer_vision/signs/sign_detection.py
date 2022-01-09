import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from ..tracking import Tracker
from tensorflow.keras.utils import to_categorical

# Globals
image_number = 0
is_model_loaded = False
model = 0
sign_classes = ["speed_sign_30", "speed_sign_60", "speed_sign_90", "stop_sign", "left_turn", "sign_not_found"]
sign_tracker = Tracker()

def image_for_model(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # model requires rgb format
    image = cv2.resize(image,(30,30)) # resize image
    image = np.expand_dims(image, axis=0)
    return image

def find_sign(frame, frame_cropped):
    # Hough circles function requires greyscale image
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if (sign_tracker.mode == "detection"):
        # Use HoughCircles OpenCV to find circles in frame
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        circles = cv2.HoughCircles(frame_grey, cv2.HOUGH_GRADIENT, 1, 100, param1=250, param2=30, minRadius=20, maxRadius=100)

        if circles is not None:
            circles = np.uint16(np.around(circles)) # convert to unsigned int16

            # Get centre/radius of each circle found
            for i in circles[0,:]:
                centre = (i[0], i[1])
                radius = i[2] + 5

                # Get Region Of Interest (ROI) from each circle
                try:
                    # Get image sign from frame and display in new window
                    circle_start_point = (centre[0] - radius, centre[1] - radius)
                    circle_end_point = (centre[0] + radius, centre[1] + radius)
                    sign_image = frame[circle_start_point[1]:circle_end_point[1], circle_start_point[0]:circle_end_point[0]]
                    cv2.imshow("ROI", sign_image)
                    
                    # Run sign through CNN model and give sign category
                    current_sign_category = sign_classes[np.argmax(model(image_for_model(sign_image)))]

                    # Perform tracking of the sign detected
                    if(current_sign_category != "sign_not_found"): # ensure circle detected was a sign
                        # Show classification in camera window and circle sign to signal sign detection       
                        cv2.putText(frame_cropped, current_sign_category, (circle_end_point[0]-80, circle_start_point[1]-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                        cv2.circle(frame_cropped, (i[0], i[1]), i[2], (0,255,0), 2) # draw the outer circle

                        # Use confidence value to determine the action to take
                        match_found, match_idx = sign_tracker.confidence_matching(centre)
                        if match_found:
                            sign_tracker.known_centres_confidence[match_idx] += 1
                            sign_tracker.known_centres_classes_confidence[match_idx][sign_classes.index(current_sign_category)] += 1
                            
                            # If same sign detected at least 3 times, update tracker
                            if(sign_tracker.known_centres_confidence[match_idx] > 2):
                                max_value = max(sign_tracker.known_centres_classes_confidence[match_idx])
                                max_index = sign_tracker.known_centres_classes_confidence[match_idx].index(max_value)
                                print("Sign ahead estimation: " + sign_classes[max_index]) # debug
                                sign_tracker.get_tracking_points(sign_classes[max_index], frame_grey, frame_cropped, circle_start_point, circle_end_point)
                        
                        # Otherwise if sign was detected first time, update signs location and its detected count
                        else:
                            sign_tracker.known_centres.append(centre)
                            sign_tracker.known_centres_confidence.append(1)
                            sign_tracker.known_centres_classes_confidence.append(list(to_categorical(sign_classes.index(current_sign_category), num_classes=6)))

                    # Save dataset so it can be used for training
                    if(current_sign_category =="speed_sign_30"): class_id ="0"
                    elif(current_sign_category =="speed_sign_60"): class_id ="1"
                    elif(current_sign_category =="speed_sign_90"): class_id ="2"
                    elif(current_sign_category =="stop"): class_id ="3"
                    elif(current_sign_category =="left_turn"): class_id ="4"
                    else: class_id ="5"

                    global image_number
                    image_number = image_number + 1
                    image_folder = os.path.abspath("src/self_driving_car/data/dataset/" + class_id) 
                    image_file = image_folder + "/" + str(image_number) + ".png"
                    if not os.path.exists(image_folder):
                        os.makedirs(image_folder)
                    cv2.imwrite(image_file, sign_image)

                except Exception as e:
                    print(e) # error
                    pass
            
            # Create new window showing if sign was detected or not
            cv2.circle(frame_cropped, (i[0], i[1]), i[2], (0,0,255), 1) # draw the outer circle
            cv2.circle(frame_cropped, (i[0], i[1]), 2, (0,100,255), 2) # draw the inner circle
            cv2.imshow("SIGN FOUND", frame_cropped)
    else:
        # Tracking Mode
        sign_tracker.track(frame_grey, frame_cropped)
        
def detect_signs(frame, frame_cropped):
    # Load CNN model so it is ready for use
    global is_model_loaded
    if not is_model_loaded: # load model only once
        global model
        model = load_model(os.path.join(os.getcwd(), "src/self_driving_car/data/sign_detection_model.h5"), compile=False)
        model.summary()
        is_model_loaded = True

    find_sign(frame, frame_cropped)

    return sign_tracker.mode, sign_tracker.tracked_class
