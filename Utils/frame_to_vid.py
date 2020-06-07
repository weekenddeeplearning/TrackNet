import cv2
import os

image_folder = r'C:\Users\rswai\Google_Drive\AIML\Projects\TrackNetMirror-master\Data\Tennis\game10\Clip12'
video_name = 'output.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# fourcc = cv2.cv.CV_FOURCC('MP4V')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video = cv2.VideoWriter(video_name, fourcc, 30, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()