# ========================================================================= #
#                   Object Detection on a Video with YOLOv3                 #
# ========================================================================= #

# Import dependencies
import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture(r'C:\Users\gorke\Desktop\Object Detection with YOLOv3\pretrained_video\videos\guney.mp4')
# cap = cv2.VideoCapture(0) # in order to use webcam

# Creating a while loop
while True:
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 1) # in case of using webcam we need to flip it.

    # Get the width and the height of per frame
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    # Define the blob
    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB = True, crop = False)

    # Define the labels
    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
             "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
             "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
             "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
             "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
             "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
             "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
             "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
             "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
             "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

    # Define different colors for each label
    colors = ['0, 255, 255', '0, 0, 255', '255, 0, 0', '255, 255,0', '0, 255, 0']

    # To get the per value in colors list we need to do a couple of more operations
    colors = [np.array(color.split(',')).astype('int') for color in colors]  # turn the values into integers
    colors = np.array(colors)  # get all the values into an array
    colors = np.tile(colors, (18, 1))  # we have more objects than 5, thus we need to expand our color values

    # Create the model
    model = cv2.dnn.readNetFromDarknet(r"C:\Users\gorke\Desktop\Object Detection with YOLOv3\pretrained_model\yolov3.cfg",
                                       r"C:\Users\gorke\Desktop\Object Detection with YOLOv3\pretrained_model\yolov3.weights")

    # Get all the layers
    layers = model.getLayerNames()

    # Get output layers
    output_layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()]

    # Set the input layer
    model.setInput(frame_blob)

    # Define detection layers
    detection_layers = model.forward(output_layer)

    # Creating lists for non-maximum suppression
    ids = []
    boxes = []
    confidences = []

    for detection_layer in detection_layers:
        for object_detection in detection_layer:

            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]

            if confidence > 0.20:

                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype('int')

                start_x, start_y = int(box_center_x - (box_width / 2)), int(box_center_y - (box_height / 2))

                ids.append(predicted_id)
                confidences.append(float(confidence))
                boxes.append([start_x, start_y, int(box_width), int(box_height)])

    max_ids = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for max_id in max_ids:

        max_class_id = max_id
        box = boxes[max_class_id]

        start_x, start_y, box_width, box_height = box[0], box[1], box[2], box[3]

        predicted_id = ids[max_class_id]
        label = labels[predicted_id]
        confidence = confidences[max_class_id]

        end_x, end_y = start_x + box_width, start_y + box_height

        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]

        label = '{}: {:.2f}%'.format(label, confidence * 100)
        print('Predicted Object {}'.format(label))

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 1)
        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # Show the video
    cv2.imshow('Detection Window', frame)

    # Keep the video open till we want to close it, if we need to close it then we'll use q from the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()