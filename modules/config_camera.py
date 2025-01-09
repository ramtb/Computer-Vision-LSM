import cv2

class CameraHandler:
    def __init__(self, camera_index=0, width_screen=1280, height_screen=720):
        """
        Initializes the CameraHandler object.

        Parameters:
        - camera_index (int): Index of the camera (0 for default, 1 for external).
        - width_screen (int): Desired width of the camera frame.
        - height_screen (int): Desired height of the camera frame.
        """
        self.camera_index = camera_index
        self.width_screen = width_screen
        self.height_screen = height_screen
        self.cap = cv2.VideoCapture(self.camera_index)
        
        

    def set_resolution(self, width, height):
        """
        Sets the resolution of the camera.

        Parameters:
        - width (int): Desired frame width.
        - height (int): Desired frame height.
        """
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_resolution(self):
        """
        Gets the current resolution of the camera.

        Returns:
        - tuple: (width, height) of the current camera resolution.
        """
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def capture_frames(self):
        """
        Captures video frames in real time and displays them.

        Press 'q' to exit the video capture loop.
        """
        print("Press 'q' to quit.")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            cv2.imshow("Camera Feed", frame)
            
            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_camera()

    def release_camera(self):
        """
        Releases the camera and closes all OpenCV windows.
        """
        self.cap.release()
        cv2.destroyAllWindows()
        
    def get_frames(self):
        """
        Get frames from the camera.
        """
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame.")
        return ret,frame
        

# Example usage
if __name__ == "__main__":
    camera = CameraHandler(camera_index=0, width_screen=1280, height_screen=720)
    width, height = camera.get_resolution()
    print("Camera Resolution:", width, height)
    camera.set_resolution(camera.width_screen, camera.height_screen)
    print('Window resolution:', camera.width_screen, camera.height_screen)
    ret, frames = camera.get_frames()
    print('ret:', ret, 'frames:', frames.shape)
    camera.capture_frames()
