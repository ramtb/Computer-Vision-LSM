import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, camera, hands):
        """
        Initializes the HandDetector object.

        Parameters:
        - camera (CameraHandler): The camera object.
        - hands (mediapipe Hands): Mediapipe hands solution object.
        """
        self.camera = camera
        self.hands = hands

    def process_frame(self, frame):
        """
    Processes a single frame to detect hand landmarks, calculates bounding box-like coordinates,
    and returns separate x, y, and z coordinates for the hand landmarks.

    Parameters:
    - frame: The video frame captured from the camera.

    Returns:
    - frame: The video frame with the landmarks drawn and connected.
    - x_positions (list): List of x-coordinate positions of the hand landmarks normalized to [0, 1].
    - y_positions (list): List of y-coordinate positions of the hand landmarks normalized to [0, 1].
    - z_positions (list): List of z-coordinate positions of the hand landmarks normalized to [0, 1].
    - flag_hands (bool): True if hands are detected, False otherwise.
        """
        x_positions = []
        y_positions = []
        z_positions = []
        flag_hands = False

        # Convert frame to RGB for Mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hands
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            # Only use the first detected hand (you can adjust this if needed for multiple hands)
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            
            # Extract x, y, z coordinates
            for landmark in hand_landmarks.landmark:
                x_positions.append(landmark.x)
                y_positions.append(landmark.y)
                z_positions.append(landmark.z)

            # Calculate bounding box coordinates (min/max of x and y positions)
            min_x, min_y = np.min(x_positions) * self.camera.width_screen, np.min(y_positions) * self.camera.height_screen
            max_x, max_y = np.max(x_positions) * self.camera.width_screen, np.max(y_positions) * self.camera.height_screen

            # Draw the bounding box around the hand
            cv2.rectangle(frame, (int(min_x), int(min_y)),
                        (int(max_x), int(max_y)), (255, 0, 0), 2)
            flag_hands = True

        return frame, x_positions, y_positions, z_positions, flag_hands


    def start_detection(self):
        """
        Starts the camera and runs hand detection in real-time.
        """
        print("Press 'q' to quit.")
        while True:
            ret, frame = self.camera.get_frames()
            # Process the frame for hand landmarks and connections
            frame, hand_positions, flag_hands = self.process_frame(frame)

            # Display the processed frame
            cv2.imshow("Hand Detection", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Thank you for using the hand detector!')
                break

        self.release_resources()

    def release_resources(self):
        """
        Releases camera resources and closes OpenCV windows.
        """
        self.camera.release_camera()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    ##### Config camera ####
    from modules.config_camera import CameraHandler
    
    camera = CameraHandler(camera_index=1, width_screen=1280, height_screen=720)  # Default camera
    width, height = camera.get_resolution()  # Get the resolution of the camera
    camera.set_resolution(camera.width_screen, camera.height_screen)  # Set the resolution of the camera window

    # Initialize Mediapipe Hands solution
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    detector = HandDetector(camera=camera, hands=hands)
    detector.start_detection()
