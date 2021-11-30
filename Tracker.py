from types import DynamicClassAttribute
import numpy as np

class Tracker():
    lastLoc = None
    dDists = np.array([[0, 0]] * 4, dtype = np.float64)

    LEFT_CLICK_THRESHOLD = 0.05
    RIGHT_CLICK_THRESHOLD = 0.07 # Slightly larger right click gap. Model has issue isolating index and middle finger for right click gesture. 
    FIST_THRESHOLD = (np.pi / 2)
    MVMT_LANDMARKS = []


    def __init__(self, HAND_LANDMARK, image_shape):
        self.HAND_LANDMARK = HAND_LANDMARK
        self.image_shape = image_shape
        self.MVMT_LANDMARKS = [
            HAND_LANDMARK.WRIST, 
            # HAND_LANDMARK.THUMB_CMC, 
            # HAND_LANDMARK.INDEX_FINGER_MCP, 
            HAND_LANDMARK.RING_FINGER_MCP, 
            HAND_LANDMARK.PINKY_MCP
        ]


    def gestureRecognition(self, hand_landmarks):
        print(hand_landmarks.landmark[self.HAND_LANDMARK.INDEX_FINGER_TIP].z, "\t", hand_landmarks.landmark[self.HAND_LANDMARK.MIDDLE_FINGER_TIP].z)
        # z coordinate is not reliable. It is reliable reliable when palm tilts along with finger. Finger-only z-coordinate change is not reliable. 

        # First check if is fist, no action if fist. 
        # A fist is the pinky folded in. Index and middle fingers gives lots of false positives for fist detection when left or right clicking. 
        v1 = np.array([
            hand_landmarks.landmark[self.HAND_LANDMARK.PINKY_MCP].x - hand_landmarks.landmark[self.HAND_LANDMARK.PINKY_PIP].x,
            hand_landmarks.landmark[self.HAND_LANDMARK.PINKY_MCP].y - hand_landmarks.landmark[self.HAND_LANDMARK.PINKY_PIP].y
        ])
        v2 = np.array([
            hand_landmarks.landmark[self.HAND_LANDMARK.PINKY_TIP].x - hand_landmarks.landmark[self.HAND_LANDMARK.PINKY_DIP].x,
            hand_landmarks.landmark[self.HAND_LANDMARK.PINKY_TIP].y - hand_landmarks.landmark[self.HAND_LANDMARK.PINKY_DIP].y
        ])
        angle = np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))

        is_fist = angle < self.FIST_THRESHOLD
        if (is_fist):
            self.reset()
            return True, False, False, np.array([0, 0], dtype = np.float64)


        # Left click is based on the distance between index tip and thumb tip. 
        index_thumb_dist = np.sqrt(
            (hand_landmarks.landmark[self.HAND_LANDMARK.INDEX_FINGER_TIP].x - hand_landmarks.landmark[self.HAND_LANDMARK.THUMB_TIP].x) ** 2 + 
            (hand_landmarks.landmark[self.HAND_LANDMARK.INDEX_FINGER_TIP].y - hand_landmarks.landmark[self.HAND_LANDMARK.THUMB_TIP].y) ** 2
        )
        is_left_click = index_thumb_dist < self.LEFT_CLICK_THRESHOLD

        # Right click is based on the distance between middle tip and thumb tip. 
        middle_thumb_dist = np.sqrt(
            (hand_landmarks.landmark[self.HAND_LANDMARK.MIDDLE_FINGER_TIP].x - hand_landmarks.landmark[self.HAND_LANDMARK.THUMB_TIP].x) ** 2 + 
            (hand_landmarks.landmark[self.HAND_LANDMARK.MIDDLE_FINGER_TIP].y - hand_landmarks.landmark[self.HAND_LANDMARK.THUMB_TIP].y) ** 2
        )
        is_right_click = middle_thumb_dist < self.RIGHT_CLICK_THRESHOLD

        # Use spacial averaging of multiple landmarks to smooth out fluctuations. 
        mouseLoc = np.array([0, 0], dtype = np.float64)
        for lmk in self.MVMT_LANDMARKS:
            mouseLoc[0] += -hand_landmarks.landmark[lmk].x
            mouseLoc[1] += hand_landmarks.landmark[lmk].y
        mouseLoc /= len(self.MVMT_LANDMARKS)
        mouseLoc[0] *= self.image_shape[1] # x is number of cols. 
        mouseLoc[1] *= self.image_shape[0]

        # Use temporal averaging of multiple timesteps to smooth out fluctuations. 
        if self.lastLoc is not None:
            self.dDists[:-1] = self.dDists[1:]
            self.dDists[-1] = mouseLoc - self.lastLoc
        self.lastLoc = mouseLoc

        return False, is_left_click, is_right_click, np.mean(self.dDists, axis = 0)


    def reset(self):
        self.dDists *= 0 # Set the vector to 0. 
        self.lastLoc = None
        