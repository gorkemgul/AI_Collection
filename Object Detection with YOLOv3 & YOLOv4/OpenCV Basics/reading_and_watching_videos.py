# ========================================================================= #
#                         Reading and Watching Videos                       #
# ========================================================================= #

# Import OpenCV
import cv2

#Getting a video from webcam
cap = cv2.VideoCapture(0) # 0 for webcam

# We need to create a while loop to read videos in OpenCV
while True:
    ret, frame = cap.read() # if cap.read is working correctly than ret will be equal True otherwise it's gonna be False.
    frame = cv2.flip(frame, 1) # we need to flip our images before we show it. 1 means flip it according to y-axis.
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'): # Each frame will be shown for 30 seconds.
        break

# Before we start to work on our video, we need to stop it.
cap.release()

# Working with a recorded video
cap = cv2.VideoCapture('music_in_the_rain.mp4')

# Create another while loop
while True:
    ret, frame = cap.read()
    if ret == 0: # we need to break our program because of the video ends which means it's gonna turn an error because ret will be 0.
        break
    frame = cv2.flip(frame, 1)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'): # Each frame will be shown for 30 seconds, and when we click q it's gonna stop our algorithm.
        break

# Before we start to work on our video, we need to stop it.
cap.release()