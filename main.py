import cv2
import logging
from pythonosc import udp_client
from threading import Thread
from gesture_control_2 import EnhancedHandGestureModel
from posture_control import EnhancedPostureModel

# Initialize logging
logging.basicConfig(filename='enhanced_recognition.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
# Initialize logging
logging.basicConfig(filename='enhanced_recognition.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Define OSC client parameters
OSC_IP = "127.0.0.1"
OSC_PORT = 4560

def process_frames(hand_model, posture_model):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logging.error("Failed to open camera.")
        return

    while cap.isOpened():
        try:
            success, image = cap.read()
            if not success:
                logging.warning("Ignoring empty camera frame.")
                continue

            # Convert image to RGB format
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Flip the image
            image_rgb = cv2.flip(image_rgb, 1)

            # Process frames with the hand gesture model
            processed_hand_image = hand_model.process_frame(image_rgb.copy())
            # Process frames with the posture model
            processed_posture_image = posture_model.process_frame(image_rgb.copy())

            # Display processed frames
            cv2.imshow('Enhanced Gesture Recognition', processed_hand_image)
            cv2.imshow('Enhanced Posture Recognition', processed_posture_image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        except Exception as e:
            logging.error(f"Error in processing loop: {e}")
    
    cap.release()
    cv2.destroyAllWindows()
    hand_model.close()
    posture_model.close()

def main():
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    hand_model = EnhancedHandGestureModel(osc_client)
    posture_model = EnhancedPostureModel(osc_client)

    process_frames(hand_model, posture_model)

if __name__ == "__main__":
    main()