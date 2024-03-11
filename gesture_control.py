import cv2
import mediapipe as mp
import logging
import numpy as np
from pythonosc import udp_client

# Initialize logging
logging.basicConfig(filename='gesture_face_recognition.log', level=logging.INFO)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Gesture thresholds
THUMB_INDEX_DISTANCE_THRESHOLD = 0.1
VICTORY_SIGN_Z_THRESHOLD = 0
THUMBS_UP_Y_THRESHOLD = 0
THUMBS_DOWN_Y_THRESHOLD = 0
CLOSED_FIST_Y_THRESHOLD = 0
POINTING_UP_Y_THRESHOLD = 0
I_LOVE_YOU_X_THRESHOLD = 0

# Gesture history
gesture_history = []

# Initialize Sonic Pi OSC client
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 4560)  # Change IP and port accordingly

# Function to send Sonic Pi OSC messages
def send_sonic_pi_osc_message(message, gesture):
    osc_client.send_message("/gesture", [message, gesture])


# Define postures based on detected gestures
def apply_posture(gesture):
    if gesture == "Open Hand Detected":
        send_sonic_pi_osc_message("C3",gesture)
    elif gesture == "Victory Sign Detected":
        send_sonic_pi_osc_message("E3",gesture)
    elif gesture == "Thumbs Up Detected":
        send_sonic_pi_osc_message("G3",gesture)
    elif gesture == "Thumbs Down Detected":
        send_sonic_pi_osc_message("A3",gesture)
    elif gesture == "Closed Fist Detected":
        send_sonic_pi_osc_message("B3",gesture)
    elif gesture == "Pointing Up Detected":
        send_sonic_pi_osc_message("C4",gesture)
    elif gesture == "I Love You Gesture Detected":
        send_sonic_pi_osc_message("D4",gesture)

def calculate_angle(a, b, c):
    """Calculate the angle between three points a, b, c where b is the vertex."""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])

    # Calculate the angle using dot product and arccosine
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Ensure the value is within valid range for arccosine
    if np.abs(cosine_angle) <= 1:
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    else:
        return None  # Return None if angle calculation is not valid

def is_hand_open(hand_landmarks):
    """Determine if the hand is open by checking the angles between fingers."""
    # Assume all fingers are extended until proven otherwise
    is_open = True

    # Define landmarks for base and tips of fingers
    finger_bases = [mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.INDEX_FINGER_MCP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP,
                    mp_hands.HandLandmark.PINKY_MCP]
    finger_tips = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]

    # Check angles between base and tip of each finger
    for base, tip in zip(finger_bases, finger_tips):
        base_landmark = hand_landmarks.landmark[base]
        tip_landmark = hand_landmarks.landmark[tip]
        next_to_base = hand_landmarks.landmark[base + 1]  # Get the landmark next to the base towards the tip

        # Calculate angle
        angle = calculate_angle(base_landmark, next_to_base, tip_landmark)
        
        # Check if the finger is not extended
        if angle < 160:  # Angle threshold to consider a finger as not extended
            is_open = False
            break

    return is_open

# Updated is_thumbs_up function
def is_thumbs_up(hand_landmarks):
    try:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Check if thumb is extended and index finger is raised above the thumb
        return (thumb_tip.y < index_mcp.y and thumb_tip.x < index_tip.x - THUMBS_UP_Y_THRESHOLD and
                calculate_angle(thumb_tip, index_mcp, index_tip) < 90)
    except Exception as e:
        logging.error(f"Error in thumbs up detection: {e}")

# Updated is_thumbs_down function
def is_thumbs_down(hand_landmarks):
    try:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Check if thumb is extended and index finger is below the thumb
        return (thumb_tip.y > index_mcp.y and thumb_tip.x < index_tip.x - THUMBS_DOWN_Y_THRESHOLD and
                calculate_angle(thumb_tip, index_mcp, index_tip) < 90)
    except Exception as e:
        logging.error(f"Error in thumbs down detection: {e}")

# Updated is_victory_sign function
def is_victory_sign(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_folded = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y < \
                  hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_folded = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y < \
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    # Check if index finger is above the middle finger and other fingers are folded
    return (index_tip.y < middle_tip.y and ring_folded and pinky_folded and
            calculate_angle(index_tip, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                            middle_tip) < 90)

# Updated is_pointing_up function
def is_pointing_up(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    other_fingers_folded = True

    for finger in [mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]:
        if hand_landmarks.landmark[finger].y < hand_landmarks.landmark[finger - 2].y + POINTING_UP_Y_THRESHOLD:
            other_fingers_folded = False
            break

    # Check if index finger is raised and other fingers are folded
    return (index_tip.y < index_mcp.y and other_fingers_folded and
            calculate_angle(index_tip, index_mcp, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]) > 160)

# Updated is_closed_fist function
def is_closed_fist(hand_landmarks):
    fingers_folded = True
    finger_tips = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]

    for tip in finger_tips:
        tip_landmark = hand_landmarks.landmark[tip]
        if tip_landmark.y < hand_landmarks.landmark[tip - 2].y + CLOSED_FIST_Y_THRESHOLD:
            fingers_folded = False
            break

    return fingers_folded

# Updated is_i_love_you function
def is_i_love_you(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    middle_folded = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y < \
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_folded = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y < \
                  hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y

    # Check if middle and ring fingers are extended and index and pinky fingers are raised
    return (not middle_folded and not ring_folded and index_tip.y < pinky_tip.y and
            thumb_tip.x < index_tip.x < pinky_tip.x - I_LOVE_YOU_X_THRESHOLD)


# Calculate the centroid of a hand
def calculate_hand_centroid(hand_landmarks, image_width, image_height):
    xs = [landmark.x * image_width for landmark in hand_landmarks.landmark]
    ys = [landmark.y * image_height for landmark in hand_landmarks.landmark]
    centroid_x, centroid_y = np.mean(xs), np.mean(ys)
    return int(centroid_x), int(centroid_y)

# Calculate the centroid of a face bounding box
def calculate_face_centroid(detection, image_width, image_height):
    bbox = detection.location_data.relative_bounding_box
    cx = int((bbox.xmin + bbox.width / 2) * image_width)
    cy = int((bbox.ymin + bbox.height / 2) * image_height)
    return cx, cy

# Function to determine if the hand is open and count the raised fingers
def count_raised_fingers(hand_landmarks):
    raised_fingers = 0
    try:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # Check Euclidean distances between finger tips and hand center
        distances = [
            ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5,
            ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5,
            ((thumb_tip.x - ring_tip.x) ** 2 + (thumb_tip.y - ring_tip.y) ** 2) ** 0.5,
            ((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2) ** 0.5
        ]

        # Count raised fingers
        for dist in distances:
            if dist > 0.1:
                raised_fingers += 1

        return raised_fingers, raised_fingers > 0
    except Exception as e:
        logging.error(f"Error in hand open detection: {e}")
        return raised_fingers, False

# Function to estimate distance based on face bounding box size
def estimate_distance_and_display(image, bbox, image_width, image_height):
    # Estimate the distance based on the size of the bounding box.
    # The exact formula here can be adjusted based on experimentation.
    # This is a simple heuristic that assumes larger face bounding boxes are closer.
    box_area = bbox[2] * bbox[3]
    normalized_area = box_area / (image_width * image_height)
    
    # Simple heuristic to convert area into a "distance" value. Adjust the scaling factor as needed.
    distance_estimate = 1 / (normalized_area + 0.1)
    
    # Display the estimated distance on the screen
    cv2.putText(image, f"Distance: {distance_estimate:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Start capturing video from the first webcam
cap = cv2.VideoCapture(0)
last_detected_gesture = None

face_color = (10, 200, 10)  # Green
hand_color = (10, 10, 200)  # Red
pose_color = (200, 200, 10)  # Yellow
distance_text_color = (255, 255, 255)  # White

while cap.isOpened():
    try:
        success, image = cap.read()
        if not success:
            break

        # Flip the image horizontally for mirror mode
        image = cv2.flip(image, 1)

        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        image.flags.writeable = False

        # Process the image and find hands
        results = hands.process(image)

        # Process the image and find faces
        face_results = face_detection.process(image)

        # Process the image and find pose
        pose_results = pose.process(image)

        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_gesture = None

        if current_gesture:
            cv2.putText(image, current_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw face detection annotations on the image
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * image_width), int(bboxC.ymin * image_height), \
                    int(bboxC.width * image_width), int(bboxC.height * image_height)
                cv2.rectangle(image, bbox, (0, 255, 0), 2)
                # Call the new function to estimate distance and display it
                estimate_distance_and_display(image, bbox, image_width, image_height)
                face_centroid = calculate_face_centroid(detection, image_width, image_height)

                if results.multi_hand_landmarks:
                    # for hand_landmarks in results.multi_hand_landmarks:
                    #     hand_centroid = calculate_hand_centroid(hand_landmarks, image_width, image_height)
                    #     cv2.line(image, hand_centroid, face_centroid, (255, 0, 0), 2)
                    for hand_landmarks in results.multi_hand_landmarks:
                        for landmark in hand_landmarks.landmark:
                            x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Count raised fingers and check if hand is open
                raised_fingers_count, is_open = count_raised_fingers(hand_landmarks)
                
                # Gesture recognition with updated logic
                if is_open:
                    current_gesture = "Open Hand Detected"
                elif is_victory_sign(hand_landmarks):
                    current_gesture = "Victory Sign Detected"
                elif is_thumbs_up(hand_landmarks):
                    current_gesture = "Thumbs Up Detected"
                elif is_thumbs_down(hand_landmarks):
                    current_gesture = "Thumbs Down Detected"
                elif is_closed_fist(hand_landmarks):
                    current_gesture = "Closed Fist Detected"
                elif is_pointing_up(hand_landmarks):
                    current_gesture = "Pointing Up Detected"
                elif is_i_love_you(hand_landmarks):
                    current_gesture = "I Love You Gesture Detected"

                # Add gesture to history
                if current_gesture:
                    gesture_history.append(current_gesture)
                    # Limit history to last 10 gestures
                    if len(gesture_history) > 10:
                        gesture_history.pop(0)

                # Print the current gesture if it's different from the last detected gesture
                if current_gesture and current_gesture != last_detected_gesture:
                    print(f"{current_gesture}")
                    last_detected_gesture = current_gesture
                    print(f"Number of raised fingers: {raised_fingers_count}\n")

                # Apply posture based on detected gesture
                if current_gesture:
                    apply_posture(current_gesture)

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the image
        cv2.imshow('Gesture and Face Recognition', image)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    except Exception as e:
        logging.error(f"Error in processing loop: {e}")

# Release resources
hands.close()
face_detection.close()
pose.close()
cap.release()
cv2.destroyAllWindows()