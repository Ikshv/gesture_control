import cv2
import mediapipe as mp
import logging
import numpy as np
from pythonosc import udp_client

class EnhancedHandGestureModel:
    def __init__(self, osc_client):
        # Initialize MediaPipe Hands with more options
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.osc_client = osc_client
        self.previous_gesture = None  # Track previous gesture

    def identify_gesture(self, hand_landmarks, handedness):
        # Example: Identify different gestures based on thumb and index finger positions
        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
        pinky_base = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]
        palm_base = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]

        # Calculate the distance between thumb tip and index finger tip
        thumb_index_distance = np.sqrt((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2)

        # Calculate distances for fist detection
        distances_from_palm = []
        for finger_tip in [index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_tip]:
            distance = np.sqrt((finger_tip.x - palm_base.x) ** 2 + (finger_tip.y - palm_base.y) ** 2)
            distances_from_palm.append(distance)

        # Check for specific gestures based on hand landmarks
        if thumb_index_distance < 0.05:  # Threshold value might need adjustment
            gesture = "thumb_index_touch"
        elif all(distance < 0.1 for distance in distances_from_palm):  # Threshold value might need adjustment
            gesture = "fist"
        elif pinky_tip.y < pinky_base.y and middle_finger_tip.y > pinky_base.y:
            gesture = "pinky_up"
        elif index_finger_tip.y < thumb_tip.y:
            gesture = "index_up"
        else:
            gesture = None

        # Append handedness to the gesture
        if handedness == "Right":
            gesture += "_right"
        elif handedness == "Left":
            gesture += "_left"

        return gesture

    def process_frame(self, image):
        # Resize the image to reduce processing time
        scale_percent = 50  # example: reduce size by 50%
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        
        results = self.mp_hands.process(resized_image)
        image.flags.writeable = True
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Optionally, you can still calculate distances or identify gestures here but consider optimizing this part

        return image


    def calculate_hand_distance(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        palm_base = landmarks[mp.solutions.hands.HandLandmark.WRIST]

        # Calculate distances from palm base to other finger tips
        distances = []
        for finger_tip in [landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP],
                           landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP],
                           landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP],
                           landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP]]:
            distance = np.sqrt((finger_tip.x - palm_base.x) ** 2 + (finger_tip.y - palm_base.y) ** 2)
            distances.append(distance)

        # Return the average distance
        return np.mean(distances)

    def close(self):
        self.mp_hands.close()
