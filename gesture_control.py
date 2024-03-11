import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to determine if the hand is open (refactored)
def is_hand_open(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

    # Check Euclidean distances between key points
    thumb_pinky_dist = ((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2) ** 0.5
    middle_ring_dist = ((middle_finger_tip.x - ring_finger_tip.x) ** 2 + (middle_finger_tip.y - ring_finger_tip.y) ** 2) ** 0.5

    return thumb_pinky_dist > 0.1 and middle_ring_dist > 0.1

def is_victory_sign(hand_landmarks):
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    y_threshold = 0.1  # Adjust as needed

    if index_finger_tip.y < ring_finger_mcp.y - y_threshold and middle_finger_tip.y < ring_finger_mcp.y - y_threshold:
        return True
    return False

def is_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    if thumb_tip.y < index_finger_mcp.y:
        return True
    return False

def is_closed_fist(hand_landmarks):
    #  Logic: For a closed fist, finger tips (except the thumb) should be closer 
    #  to the base of their corresponding fingers (MCP joints) 
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    return (
        index_finger_tip.y > index_finger_mcp.y and
        middle_finger_tip.y > middle_finger_mcp.y and
        ring_finger_tip.y > ring_finger_mcp.y and
        pinky_finger_tip.y > pinky_finger_mcp.y 
    ) 

def is_pointing_up(hand_landmarks):
    # Logic: Index finger extended, others likely curled. Check if the index 
    # fingertip is significantly higher than the index finger base.
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    
    return index_finger_tip.y < index_finger_base.y 

def is_thumbs_down(hand_landmarks):
    # Logic: Thumb pointing down, check if the thumb tip is significantly 
    # lower than the thumb's base joint (CMC)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC] 

    return thumb_tip.y > thumb_cmc.y

def is_i_love_you(hand_landmarks):
    # Logic: Index, pinky extended, thumb extended to the side, others curled.
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]  

    return (
        index_finger_tip.y < index_finger_mcp.y and 
        pinky_finger_tip.y < pinky_finger_mcp.y and 
        thumb_tip.x < thumb_ip.x  # Check if thumb is extended sideways
    )

# Start capturing video from the first webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image and find hands
    results = hands.process(image)

    # Draw the hand annotations on the image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Drawing
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Gesture recognition
            if is_hand_open(hand_landmarks):
                print("Open Hand Detected")
            elif is_victory_sign(hand_landmarks):
                print("Victory Sign Detected")
            elif is_thumbs_up(hand_landmarks):
                print("Thumbs Up Detected")
            else:
                print("Closed Fist Detected")

    # Display the image
    cv2.imshow('Gesture Recognition', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
hands.close()
cap.release()
cv2.destroyAllWindows()
