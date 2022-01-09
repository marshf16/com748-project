import cv2
from computer_vision.lanes.lane_detection import detect_lanes
from computer_vision.signs.sign_detection import detect_signs
from numpy import interp

class Control():
    def __init__(self):
        # Set initial car values at beginning of simulation
        self.angle = 0.0
        self.speed = 30
        self.previous_mode = "detection"

        # T Junction variables
        self.previous_mode_leftturn = "detection"
        self.left_turn_iter = 0
        self.new_turn_angle = 0
        self.left_turn_detected = False
        self.left_turn_activated = False
    
    def follow_lane(self, max_dist, distance_error, curvature, mode, tracked_class):
        # Constants
        MAX_RIGHT_TURN = 90
        MAX_LEFT_TURN = -90

        # Update car speed depending on sign found
        if((tracked_class != 0) and (self.previous_mode == "tracking") and (mode == "detection")):
            if(tracked_class == "speed_sign_30"):
                self.speed = 30
            elif(tracked_class == "speed_sign_60"):
                self.speed = 60
            elif(tracked_class == "speed_sign_90"):
                self.speed = 90
            elif(tracked_class == "stop_sign"):
                self.speed = 0
        
         # Update previous mode
        self.previous_mode = mode

        # Set turn angle to be made by the car
        car_turn_angle = 0
        if(distance_error > max_dist): # Max distance, perform max turn
            car_turn_angle = MAX_RIGHT_TURN + curvature
        elif(distance_error < (-1 * max_dist)): # Max distance, perform max turn
            car_turn_angle = MAX_LEFT_TURN + curvature
        else: # Inside max distance, perform smaller turn using interpolation function
            car_offset = interp(distance_error, [-max_dist, max_dist], [MAX_LEFT_TURN, MAX_RIGHT_TURN])
            car_turn_angle = car_offset + curvature
            
        # Ensure car cannot go over max turn
        if(car_turn_angle > MAX_RIGHT_TURN):
            car_turn_angle =  MAX_RIGHT_TURN
        elif(car_turn_angle < MAX_LEFT_TURN):
            car_turn_angle = MAX_LEFT_TURN

        # Update car angle
        self.angle = interp(car_turn_angle, [MAX_LEFT_TURN, MAX_RIGHT_TURN], [-45,45])

    def perform_left_turn(self, mode):
        self.speed = 50 # assigned car turning speed

        # Car starts tracking left turn...
        if((self.previous_mode_leftturn == "detection") and (mode == "tracking")): # left turn sign found
            self.previous_mode_leftturn = "tracking" # update left turn mode
            self.left_turn_detected = True 
        elif((self.previous_mode_leftturn == "tracking") and (mode == "detection")): # as soon as sign out of frame, mode returns to detection, now start the turn
            self.left_turn_detected = False
            self.left_turn_activated = True
            
            # Every 20th iteration, slighty turn the car (realism effect)
            if (((self.left_turn_iter % 20) == 0) and (self.left_turn_iter > 100) ):
                self.new_turn_angle = self.new_turn_angle - 7
            
            # After specific time, stop turn by resetting values
            if(self.left_turn_iter == 250):
                self.previous_mode_leftturn = "detection"
                self.left_turn_activated = False
                self.left_turn_iter = 0
                
            self.left_turn_iter = self.left_turn_iter + 1

        # Update car angle
        if (self.left_turn_activated or self.left_turn_detected):
            self.angle = self.new_turn_angle

    def drive(self, current_state):
        # Get important values from car class to be updated
        [distance_error, curvature, img, mode, tracked_class] = current_state

        # Make car move by following lane
        if((distance_error != 1000) and (curvature != 1000)): 
            self.follow_lane(img.shape[1]/4, distance_error, curvature, mode, tracked_class) # max distance will be quarter of the frame
        else:
            self.speed = 0.0
        
        # Left turn sign found, perform turn
        if (tracked_class == "left_turn"):
            self.perform_left_turn(mode)

        # Interpolate the angle from 'real' world to 'simulated' world
        angle_for_sim = interp(self.angle, [-45,45], [0.5,-0.5]) # The simulated car angle range is 1 (-0.5 to 0.5)

        # Interpolate the speed from 'real' world to 'simulated' world
        if (self.speed != 0): # If car is moving
            speed_for_sim = interp(self.speed, [30,90] ,[1,2]) # The simulated car speed range is 1 (1 to 2)
        else:
            speed_for_sim = 0.0

        return angle_for_sim, speed_for_sim

class Car():
    def __init__(self):
        self.Control = Control()

    def display_state(self, frame_disp, current_angle, current_speed, tracked_class):
        # Get real world angle and speed for presentation
        current_angle  = interp(current_angle, [-0.5,0.5], [45,-45])
        if (current_speed != 0.0): current_speed = interp(current_speed, [1,2], [30 ,90])

        # Add car direction to window
        if (current_angle < -10): direction = "[Left]"
        elif (current_angle > 10): direction = "[Right]"
        else: direction = "[Straight]"
        if(current_speed > 0): direction = "Moving: " + direction
        else: direction = "Stopped"
        cv2.putText(frame_disp, str(direction), (20,40), cv2.FONT_HERSHEY_DUPLEX, 0.4, (50,255,50), 1)

        # Add angle and speed of car to window
        angle_speed_str = "[Angle: " + str(int(current_angle)) + "degrees] " + "[Speed: " + str(int(current_speed)) + "mph]"
        cv2.putText(frame_disp, str(angle_speed_str), (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0,0,255), 1)
        
        # Add current sign detection to window
        if (tracked_class == "left_turn"): # for left_turn, add additional information
            if (self.Control.left_turn_detected):
                tracked_class = tracked_class + " [DETECTED]"
            else:
                tracked_class = tracked_class + " [ACTIVATED]"
        cv2.putText(frame_disp, "Recent Sign: " + str(tracked_class), (20,60), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,255,255), 1)
                
    def drive_car(self, frame):
        # Crop image to speed up image processing
        frame_cropped = frame[0:640,238:1042]
        frame_cropped = cv2.resize(frame_cropped,(320,240))
        frame_original = frame_cropped.copy()
        
        # Get data required for lane assistance
        distance, curvature = detect_lanes(frame_cropped)

        # Get data required for sign detection
        mode, tracked_class = detect_signs(frame_original, frame_cropped)

        # With data retrieved from lane/sign modules, set state of the car
        current_state = [distance, curvature, frame_cropped, mode, tracked_class]
        
        # Drive the car with all our data!
        angle, speed = self.Control.drive(current_state)

        # Display data via new window
        self.display_state(frame_cropped, angle, speed, tracked_class)
        
        return angle, speed, frame_cropped