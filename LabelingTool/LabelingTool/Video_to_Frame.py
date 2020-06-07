# Program To Read video
# and Extract Frames
import cv2
import time

# Function to extract frames
def FrameCapture(path):
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0
    frame_rate = 29.97
    prev = 0
    # checks whether frames were extracted
    success = 1

    while success:
        # vidObj object calls read
        # function extract frames

        time_elapsed = time.time() - prev
        success, image = vidObj.read()

        if time_elapsed > 1. / frame_rate:
            prev = time.time()
            cv2.imwrite("C:/Users/rswai/Google_Drive/AIML/Projects/TrackNetMirror-master/Data/images/frame_%d.jpg" % count, image)

            count += 1

# Driver Code
if __name__ == '__main__':
    # Calling the function
    FrameCapture(r"C:\Users\rswai\Google_Drive\AIML\Projects\TrackNetMirror-master\Data\vids\vid1_1.mp4")
