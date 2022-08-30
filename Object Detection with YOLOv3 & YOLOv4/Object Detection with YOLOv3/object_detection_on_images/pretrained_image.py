# ========================================================================= #
#                   Object Detection on an Image with YOLOv3                #
# ========================================================================= #

# Import dependencies
import cv2
import numpy as np

# Load the image
image = cv2.imread('images/group_of_people.jpg')

# Check the values of the image
print(image)

# Check the shape of our image
print(image.shape)

# We need to get height and width of our image
image_height, image_width = image.shape[:2]

# We must change our image to blob
image_blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB = True, crop = False)
# Note: the best optimized value for scalefactor is 1/255.
# Note2: the reason why we use 416, 416 is because our model was trained on the 416x416 images.
# Note3: We need to set the parameter swapRB to True because we must change our image from BGR to RGB.

# define the labels we want into the labels list
labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
         "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
         "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
         "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
         "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
         "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
         "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
         "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
         "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
         "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

# Define different colors for each label
colors = ['0, 255, 255', '0, 0, 255', '255, 0, 0', '255, 255,0', '0, 255, 0']

# To get the per value in colors list we need to do a couple of more operations
colors = [np.array(color.split(',')).astype('int') for color in colors] # turn the values into integers
colors = np.array(colors) # get all the values into an array
colors = np.tile(colors, (18, 1)) # we have more objects than 5, thus we need to expand our color values

# Create the model
model = cv2.dnn.readNetFromDarknet(r"C:\Users\gorke\Desktop\Object Detection with YOLOv3\pretrained_model\yolov3.cfg",
                                   r"C:\Users\gorke\Desktop\Object Detection with YOLOv3\pretrained_model\yolov3.weights")

# Get the layer names of our model to do detection operation
model_layers = model.getLayerNames()

# Get the output layer
output_layer = [model_layers[layer - 1] for layer in model.getUnconnectedOutLayers()]

# Set the input layer
model.setInput(image_blob)

# Create detection layers
detection_layers = model.forward(output_layer)

# Creating a couple of lists for Non-maximum suppression
ids_list = []
boxes_list = []
confidences_list = []

# Create the detection algorithm
for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]

        if confidence > 0.20:
            label = labels[predicted_id]
            bounding_box = object_detection[0:4] * np.array([image_width, image_height, image_width, image_height])
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype('int')

            start_x = int(box_center_x - (box_width / 2))
            start_y = int(box_center_y - (box_height / 2))

            ids_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append([start_x, start_y, int(box_width), int(box_height)])

# Non-maximum suppression
max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

for max_id in max_ids:
    max_class_id = max_id
    box = boxes_list[max_class_id]

    start_x = box[0]
    start_y = box[1]
    box_width = box[2]
    box_height = box[3]

    predicted_id = ids_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidences_list[max_class_id]

    end_x = start_x + box_width
    end_y = start_y + box_height
    box_color = colors[predicted_id]
    box_color = [int(each) for each in box_color]

    label = '{}: {:.2f}%'.format(label, confidence * 100)
    print('Predicted Object {}'.format(label))
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), box_color, 1)
    cv2.putText(image, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

cv2.imshow('Detection Window', image)
cv2.waitKey(0)