import enum
from classifier.GestureLabels import GestureLabels
from inferencer.Hands import HandLandmark as HAND_LANDMARK
import numpy as np
import enum

class TrackingSource(enum.IntEnum):
  """The 21 hand landmarks."""
  HAND_LANDMARKS = 0
  PALM = 1

MVMT_LANDMARKS = [
    HAND_LANDMARK.WRIST, 
    # HAND_LANDMARK.THUMB_CMC, 
    # HAND_LANDMARK.INDEX_FINGER_MCP, 
    HAND_LANDMARK.RING_FINGER_MCP, 
    HAND_LANDMARK.PINKY_MCP
]

LAST_N_DDIST_HAND_LANDMARKS = 1 # Temporal average smoothing, 1 is disable. 
LAST_N_DDIST_PALM = 4 # Temporal average smoothing, 1 is disable. 
LEFT_CLICK_THRESHOLD = 0.05
RIGHT_CLICK_THRESHOLD = 0.07 # Slightly larger right click gap. Model has issue isolating index and middle finger for right click gesture. 
FIST_THRESHOLD = np.pi / 2

class Tracker():
    lastLoc = None


    def __init__(self, image_shape, trackWith=TrackingSource.HAND_LANDMARKS):
        self.image_shape = image_shape
        self.trackWith = trackWith
        if (self.trackWith == TrackingSource.HAND_LANDMARKS):
            self.dDists = np.array([[0, 0]] * LAST_N_DDIST_HAND_LANDMARKS, dtype = np.float64)
        else:
            self.dDists = np.array([[0, 0]] * LAST_N_DDIST_PALM, dtype = np.float64)


    def trackMovement(self, hand_landmarks, palm, gesture):
        if gesture == GestureLabels.FIST:
            self.reset()

        # Use spacial averaging of multiple landmarks to smooth out fluctuations. 
        if self.trackWith == TrackingSource.PALM:
            mouseLoc = np.array([-palm.x_center, palm.y_center], dtype = np.float64)
        else:
            mouseLoc = np.array([0, 0], dtype = np.float64)
            for lmk in MVMT_LANDMARKS:
                mouseLoc[0] += -hand_landmarks.landmark[lmk].x
                mouseLoc[1] += hand_landmarks.landmark[lmk].y
            mouseLoc /= len(MVMT_LANDMARKS)
        mouseLoc[0] *= self.image_shape[1] # x is number of cols. 
        mouseLoc[1] *= self.image_shape[0]

        # Use temporal averaging of multiple timesteps to smooth out fluctuations. 
        if self.lastLoc is not None:
            self.dDists[:-1] = self.dDists[1:]
            self.dDists[-1] = mouseLoc - self.lastLoc
        self.lastLoc = mouseLoc
        
        dDist = np.mean(self.dDists, axis = 0)

        return dDist


    def reset(self):
        self.dDists *= 0 # Set the vector to 0. 
        self.lastLoc = None
        