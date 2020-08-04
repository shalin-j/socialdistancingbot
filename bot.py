import cv2
import numpy as np
import time
from playsound import playsound

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load Video
cap = cv2.VideoCapture(1 + cv2.CAP_DSHOW)

font = cv2.FONT_HERSHEY_DUPLEX
score = 0
last_detection = 0
frame_id = 0

while True:
    _, frame = cap.read()

    frame_id += 1

    height, width, channels = frame.shape

    # Convert Img to Blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv2.imshow(str(n), img_blob)

    # Pass Blob to Algorithm
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing Information on Screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]  # get class id
            class_id = np.argmax(scores)  # determines what object it is
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)  # values multiplied by original height and width of image
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle Coordinates
                x = int(center_x - w / 2)  # gets top left x
                y = int(center_y - h / 2)  # gets top left y

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Removing Extra Detections
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Creating Boxes and Labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detection = ""
            if label == 'person':
                color = (0, 255, 0)
                if h > 320: # experimental height value to see if person is too close
                    color = (0, 0, 255)
                    detection = ": too close"
                    if time.time() - last_detection > 5:  # gap between scoring
                        score += 1
                        last_detection = time.time()
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + detection, (x, y - 10), font, 1, (0, 0, 0), 2)

    ## cv2.putText(frame, "smh: " + str(score), (10, 30), font, 1, (255, 255, 255), 1)
    cv2.imshow("image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
