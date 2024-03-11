import cv2
import mediapipe as mp
import logging
import numpy as np
from pythonosc import udp_client

class EnhancedPostureModel:
    def __init__(self, osc_client):
        # Initialize MediaPipe Pose with more options
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.osc_client = osc_client

    def identify_posture(self, pose_landmarks):
        # Example: Identify posture based on landmark positions
        # Here you can define your own logic to identify different postures

        # Calculate shoulder width
        left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)

        # Calculate distance between left wrist and right wrist
        left_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        wrist_distance = abs(left_wrist.x - right_wrist.x)

        # Determine posture based on shoulder width and wrist distance
        if wrist_distance < shoulder_width * 0.4:
            posture = "Hands together"
        elif left_wrist.x < right_wrist.x:
            posture = "Left hand forward"
        else:
            posture = "Right hand forward"

        return posture

    def process_frame(self, image):
        results = self.mp_pose.process(image)
        if results.pose_landmarks:
            posture = self.identify_posture(results.pose_landmarks)
            if posture:
                self.osc_client.send_message("/posture", posture)  # Sending posture to OSC client

        return image

    def close(self):
        self.mp_pose.close()


# def main():
#     posture_model = EnhancedPostureModel()
#     cap = cv2.VideoCapture(0)

#     while cap.isOpened():
#         try:
#             success, image = cap.read()
#             if not success:
#                 logging.warning("Ignoring empty camera frame.")
#                 continue

#             # Convert image to RGB format
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             # Flip the image
#             image_rgb = cv2.flip(image_rgb, 1)

#             # Process the frame
#             processed_image = posture_model.process_frame(image_rgb)

#             cv2.imshow('Enhanced Posture Recognition', processed_image)
#             if cv2.waitKey(5) & 0xFF == ord('q'):
#                 break
#         except Exception as e:
#             logging.error(f"Error in processing loop: {e}")
    
#     cap.release()
#     cv2.destroyAllWindows()
#     posture_model.close()

# if __name__ == "__main__":
#     main()
