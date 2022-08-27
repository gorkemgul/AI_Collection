# ========================================================================= #
#                    Reading, Saving and Showing Images                     #
# ========================================================================= #

# Import OpenCV Library
import cv2

# Read the image
img = cv2.imread(filename = 'james_web.jpg', flags = None)
# Note: If we want to read our image in grayscale mode, we need to enter a flag like this: flags = cv2.IMREAD_GRAYSCALE, or we could also enter 0.

# Check the values of the image
print(img)

# Create a window with a suitable name and size
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

# Show our image
cv2.imshow('Image', img)
# Note: When we want to show our image if we don't use "cv2.waitKey" our images will be showed and closed immediately.
cv2.waitKey(0) # 0 means as long as we don't close our image it won't disappear

# Save the image
cv2.imwrite('saved_image.jpg', img)