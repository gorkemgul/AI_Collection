# ========================================================================= #
#                             Writing Texts                                 #
# ========================================================================= #

# Import dependencies
import cv2
import numpy as np

# Creating a canvas
canvas = np.zeros((512, 512, 3), dtype = np.uint8) + 255 # adding 255 for making the background white

# Defining a couple of different fonts
first_font = cv2.FONT_HERSHEY_SIMPLEX
second_font = cv2.FONT_HERSHEY_COMPLEX
third_font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

# Writing text
cv2.putText(canvas, 'OpenCV', (30, 100), first_font, 3, (0, 0, 0), cv2.LINE_AA)

# Showing our canvas after writing OpenCV on it
cv2.imshow('Canvas with a Text', canvas)
cv2.waitKey(0)