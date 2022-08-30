# ========================================================================= #
#                           Drawing Functions                               #
# ========================================================================= #

# Import dependencies
import cv2
import numpy as np

# Creating a canvas
canvas = np.zeros((512, 512, 3), dtype = np.uint8) # for drawing purposes we need to use uint8 type

# Check the values of our canvas
print(canvas)

# Show our canvas
cv2.imshow('Canvas', canvas)
cv2.waitKey(0) # for keeping the canvas as long as we want

# Turn the background of canvas into white color
white_bg_canvas = canvas + 255 # bgr -> (0, 0, 0). To turn it into white, we need to change its values to 255. (bgr -> (255, 255, 255))

# Check the values of our changed canvas
print(white_bg_canvas)

# Show our changed canvas
cv2.imshow('Canvas with White Background', white_bg_canvas)
cv2.waitKey(0)

# Let's start with drawing line
cv2.line(white_bg_canvas, (50,50), (512, 512), (255, 0, 0), thickness = 5)
cv2.line(white_bg_canvas, (100,50), (200, 250), (0, 0, 255), thickness = 7)

# Draw a rectangle
cv2.rectangle(white_bg_canvas, (20, 20), (50, 50), (0, 255, 0), thickness = -1) # to fill it, we need to change thickness to "-1"
cv2.rectangle(white_bg_canvas, (50, 50), (100, 100), (0, 255, 0), thickness = -1)

# Draw a circle
cv2.circle(white_bg_canvas, (250, 250), 100, (0, 0 , 255), thickness = -1)

# Draw a triangle
# Note: OpenCV doesn't have a function to draw a triangle but we can draw it by using 3 lines.
p1 = (100, 200)
p2 = (50, 50)
p3 = (300, 100)

cv2.line(white_bg_canvas, p1, p2, (0, 0, 0), 4)
cv2.line(white_bg_canvas, p2, p3, (0, 0, 0), 4)
cv2.line(white_bg_canvas, p1, p3, (0, 0, 0), 4)

# Draw an ellipse
cv2.ellipse(white_bg_canvas, (300, 300), (100, 50), 0, 0, 360, (255, 255, 0), thickness = -1)

# Lastly, let's look at how to use polyline for drawing purposes
# Create a point array
points = np.array([[110, 200], [330, 200], [290, 220], [100, 100]], dtype = np.int32)
cv2.polylines(white_bg_canvas, [points], True, (0, 0, 100), 5)

# Show our canvas
cv2.imshow('Canvas with White Background', white_bg_canvas)
cv2.waitKey(0)