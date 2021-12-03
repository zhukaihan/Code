# Modified from https://google.github.io/mediapipe/solutions/hands.html. 

import cv2
import mediapipe as mp
import numpy as np

import Hands
from Mouse import Mouse
from Tracker import Tracker, TrackingSource

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = Hands


def draw_palm_bbox(image, palm):

  palm_c = np.array([palm.x_center * image.shape[1], palm.y_center * image.shape[0]], dtype=int)
  
  halfX = palm.width / 2 * image.shape[1]
  halfY = palm.height / 2 * image.shape[0]
  R = np.array([
    [np.cos(-palm.rotation), -np.sin(-palm.rotation)],
    [np.sin(-palm.rotation), np.cos(-palm.rotation)]
  ])
  palm_ul = palm_c + (np.array([-halfX, -halfY]) @ R).astype(int)
  palm_ll = palm_c + (np.array([-halfX, halfY]) @ R).astype(int)
  palm_ur = palm_c + (np.array([halfX, -halfY]) @ R).astype(int)
  palm_lr = palm_c + (np.array([halfX, halfY]) @ R).astype(int)
  
  cv2.line(image, palm_ul, palm_ll, color=(255, 0, 0))
  cv2.line(image, palm_ll, palm_lr, color=(255, 0, 0))
  cv2.line(image, palm_lr, palm_ur, color=(255, 0, 0))
  cv2.line(image, palm_ur, palm_ul, color=(255, 0, 0))
  cv2.circle(image, palm_c, 5, color=(255, 0, 0))



# # For static images:
# IMAGE_FILES = []
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#   for idx, file in enumerate(IMAGE_FILES):
#     # Read an image, flip it around y-axis for correct handedness output (see
#     # above).
#     image = cv2.flip(cv2.imread(file), 1)
#     # Convert the BGR image to RGB before processing.
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Print handedness and draw hand landmarks on the image.
#     print('Handedness:', results.multi_handedness)
#     if not results.multi_hand_landmarks:
#       continue
#     image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       print('hand_landmarks:', hand_landmarks)
#       print(
#           f'Index finger tip coordinates: (',
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#       )
#       mp_drawing.draw_landmarks(
#           annotated_image,
#           hand_landmarks,
#           mp_hands.HAND_CONNECTIONS,
#           mp_drawing_styles.get_default_hand_landmarks_style(),
#           mp_drawing_styles.get_default_hand_connections_style())
      
#     cv2.imwrite(
#         '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
#     # Draw hand world landmarks.
#     if not results.multi_hand_world_landmarks:
#       continue
#     for hand_world_landmarks in results.multi_hand_world_landmarks:
#       mp_drawing.plot_landmarks(
#         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)




TRACKING_WITH = TrackingSource.HAND_LANDMARKS
mouse = Mouse()
cap = cv2.VideoCapture(0)
if cap.isOpened():
  success, image = cap.read()
  tracker = Tracker(image_shape=image.shape[:2], trackWith=TRACKING_WITH)

# For webcam input:
# static_image_mode allows palm bbox output, but lowers landmark quality as it no longer uses last landmark info. 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    max_num_hands=1,
    min_tracking_confidence=0.5,
    static_image_mode=(TRACKING_WITH == TrackingSource.PALM)) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      # Hand detected. 

      hand_landmarks = results.multi_hand_landmarks[0]

      # Draw hand landmarks. 
      mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

      # Draw palm's bounding box. 
      palm = None
      if results.hand_rects_from_palm_detections:
        palm = results.hand_rects_from_palm_detections[0]
        draw_palm_bbox(image, palm)

      # Detect gesture. 
      is_fist, is_left_click, is_right_click, dDist = tracker.gestureRecognition(hand_landmarks, palm)
      # Control mouse. 
      mouse(is_left_click, is_right_click, dDist)

    else:
      # No hand, reset location averaging of the tracker. 
      tracker.reset()
      
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
