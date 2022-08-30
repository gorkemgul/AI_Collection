# ========================================================================= #
#                                Trackbar                                   #
# ========================================================================= #

# Import dependencies
import cv2
import numpy as np

def nothing():
    pass

img = np.zeros((512, 512, 3), dtype = np.uint8)
cv2.namedWindow('Image')

# Creating the Trackbar
# To use createTrackbar function correctly, we must create a function called nothing
cv2.createTrackbar('R', 'Image', 0, 255, nothing) # Red
cv2.createTrackbar('G', 'Image', 0, 255, nothing) # Green
cv2.createTrackbar('B', 'Image', 0, 255, nothing) # Blue

# Create the switch
switch = '0: OFF, 1: ON'
cv2.createTrackbar(switch, 'Image', 0, 1, nothing)

# To change trackbars positions we need to create a while loop
while True:
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Get the trackbars
    r = cv2.getTrackbarPos('R', 'Image')
    g = cv2.getTrackbarPos('G', 'Image')
    b = cv2.getTrackbarPos('B', 'Image')
    s = cv2.getTrackbarPos(switch, 'Image')

    if switch == 0:
        img[:] = [0, 0, 0]

    if switch == 1:
        img[:] = [b, g, r]

    # Send the trackbars to their new positions
    img[:] = [b, g, r]