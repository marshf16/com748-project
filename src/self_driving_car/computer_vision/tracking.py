import math
import numpy as np
import cv2

class Tracker:
    def __init__(self):
        self.mode = "detection"
        self.tracked_class = 0

        self.known_centres = []
        self.known_centres_confidence = []
        self.known_centres_classes_confidence = []

        self.previous_frame_grey = 0
        self.p0 = []

        self.mask = 0
        self.color = np.random.randint(0, 255, (100, 3)) 

    def distance(self,a,b):
        return math.sqrt( ( (float(a[1])-float(b[1]))**2 ) + ( (float(a[0])-float(b[0]))**2 ) )

    def confidence_matching(self,center):
        # Set default values
        match_found = False
        match_idx = 0
        for i in range(len(self.known_centres)):
            if(self.distance(center, self.known_centres[i]) < 100): # 100 is the maximum distance between two points to be considered same sign
                match_found = True
                match_idx = i
                return match_found, match_idx
        return match_found, match_idx # no match found
        
    def get_tracking_points(self, sign_class, frame_grey, frame_cropped, p1, p2, mask_to_track=None):

        # Begin tracking
        if mask_to_track is None:
            sign_mask = np.zeros_like(frame_grey) # create frame matching original
            sign_mask[p1[1]:p2[1], p1[0]:p2[0]] = 255 # using circle points passed through function
            self.p0 = cv2.goodFeaturesToTrack(frame_grey, mask=sign_mask, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7) # track these points using cv2 goodFeaturesToTrack function
        else:
            self.p0 = cv2.goodFeaturesToTrack(frame_grey, mask=mask_to_track, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        
        self.mode = "tracking"
        self.tracked_class = sign_class
        self.previous_frame_grey = frame_grey
        self.mask = np.zeros_like(frame_cropped)

    def track(self, frame_grey, frame_cropped):
        # Use Lucas Kanade
        points, status, error = cv2.calcOpticalFlowPyrLK(self.previous_frame_grey, 
                                             frame_grey,
                                             self.p0,
                                             None, 
                                             winSize=(15, 15),
                                             maxLevel=2,
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                             10,
                                             0.03))

        # Check for points found from Lucas Kanade method
        if ((points is None) or (len(points[status == 1]) < 3)): # No points found, set car mode to detection
            self.mode = "detection"
            self.mask = np.zeros_like(frame_cropped)
            self.reset()
        else:
            new_point = points[status == 1]
            old_point = self.p0[status == 1]
            
            # Draw line between points to show tracking succeeded
            for i, (new, old) in enumerate(zip(new_point, old_point)):
                a, b = (int(x) for x in new.ravel())
                c, d = (int(x) for x in old.ravel())
                
                self.mask = cv2.line(self.mask, (a, b), (c, d), self.color[i].tolist(), 2)
                frame_cropped = cv2.circle(frame_cropped, (a, b), 5, self.color[i].tolist(), -1)

            # Update frame with new drawn lines
            frame_cropped_new = frame_cropped + self.mask
            np.copyto(frame_cropped, frame_cropped_new)
            self.previous_frame_grey = frame_grey.copy() # ensure previous frame updated with new one
            self.p0 = new_point.reshape(-1, 1, 2)

    def reset(self):
        self.known_centres = []
        self.known_centres_confidence = []
        self.known_centres_classes_confidence = []
        self.previous_frame_grey = 0
        self.p0 = []


