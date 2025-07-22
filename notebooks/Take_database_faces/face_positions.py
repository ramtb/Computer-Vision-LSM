import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    def __init__(self,camera, face_mesh):
        """
        Initializes the FaceMeshDetector object.

        Parameters:
        - camera_index (int): Index of the camera (0 for default, 1 for external).
        - width_screen (int): Desired width of the camera frame.
        - height_screen (int): Desired height of the camera frame.
        """
        # Camera setup
        self.camera = camera
        self.face_mesh = face_mesh

    def process_frame(self,frame):
        """
        Processes a single frame for face landmarks and draws a rectangle.

        Parameters:
        - frame: The video frame captured from the camera.

        Returns:
        - frame: The video frame with the rectangle drawn around the detected face.
        - positions_x (np.array): X-coordinates of the detected face landmarks normalized 0 to 1.
        - positions_y (np.array): Y-coordinates of the detected face landmarks  normalized 0 to 1.
        - flag_face (int): True if a face is detected, False otherwise.
        """
        # Convert frame to RGB for MediaPipe processing
        positions_x = None
        positions_y = None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        flag_face = False
        
        if results.multi_face_landmarks:
            
            # Only use the first detected face
            lm = results.multi_face_landmarks[0]
            landmarks = lm.landmark

            # Extract landmark positions
            positions_x = np.array([landmark.x for landmark in landmarks])
            positions_y = np.array([landmark.y for landmark in landmarks])

            # Calculate bounding rectangle coordinates
            min_x, min_y = np.min(positions_x * self.camera.width_screen), np.min(positions_y * self.camera.height_screen)
            max_x, max_y = np.max(positions_x * self.camera.width_screen), np.max(positions_y * self.camera.height_screen)

            # Draw the rectangle around the face
            cv2.rectangle(frame, (int(min_x), int(min_y)),
                          (int(max_x), int(max_y)), (255, 0, 0), 2)
            flag_face = True

        return frame, positions_x, positions_y, flag_face

    def start_detection(self):
        """
        Starts the camera and runs face detection in real-time.
        """
        print("Press 'q' to quit.")
        while True:
            ret, frame = self.camera.get_frames()
            # Process the frame for face landmarks
            positions_x, position_y,  flag_face = self.process_frame(frame)

            # Display the processed frame
            cv2.imshow("Face Mesh Detection", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Thank you for using the face mesh detector!')
                break
            # return positions_x, position_y

        self.release_resources()

    def release_resources(self):
        """
        Releases camera resources and closes OpenCV windows.
        """
        self.camera.release_camera()
        cv2.destroyAllWindows()
        self.face_mesh.close()

# Example usage
if __name__ == "__main__":
    ##### Config camera ####
    from config_camera import CameraHandler
    
    camera = CameraHandler(camera_index=0, width_screen=1280, height_screen=720) ### 0 is the default camera, 1 is the external camera
    width, height = camera.get_resolution() ### Get the resolution of the camera
    camera.set_resolution(camera.width_screen, camera.height_screen) ### Set the resolution of the window of the frame

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5)
    
    detector = FaceMeshDetector(camera=camera,face_mesh=face_mesh_images)
    detector.start_detection()




def draw_regions(frame, camera):
    # Draw the lines for four regions
    start_v_line = (int(0.5 * camera.width_screen), 0)
    final_v_line = (int(0.5 * camera.width_screen), camera.height_screen)
    start_h_line = (0, int(0.5 * camera.height_screen))
    final_h_line = (camera.width_screen, int(0.5 * camera.height_screen))
    cv2.line(frame, start_v_line, final_v_line, color=(0, 0, 255), thickness=2)
    cv2.line(frame, start_h_line, final_h_line, color=(0, 0, 255), thickness=2)

    # Add text to regions
    color = (0, 0, 0)
    thickness = 2
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2
    cv2.putText(frame, 'Region 1', (0, int(camera.height_screen * 0.49)), font, font_scale, color, thickness)
    cv2.putText(frame, 'Region 2', (int(camera.width_screen * 0.5), int(camera.height_screen * 0.49)), font, font_scale, color, thickness)
    cv2.putText(frame, 'Region 3', (0, int(camera.height_screen * 0.99)), font, font_scale, color, thickness)
    cv2.putText(frame, 'Region 4', (int(camera.width_screen * 0.5), int(camera.height_screen * 0.99)), font, font_scale, color, thickness)






