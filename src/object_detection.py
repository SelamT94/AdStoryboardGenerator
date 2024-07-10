import cv2
import numpy as np

# Load YOLO model
def load_yolo_model(cfg_path, weights_path, names_path):
    net = cv2.dnn.readNet(cfg_path, weights_path)
    with open(names_path, 'r') as f:
        classes = f.read().strip().split('\n')
    return net, classes

# Perform object detection
def detect_objects(image, net, classes):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], confidences[i], classes[class_ids[i]]) for i in indices.flatten()]

if __name__ == "__main__":
    yolo_cfg_path = "models/yolo_model/yolov3.cfg"
    yolo_weights_path = "models/yolo_model/yolov3.weights"
    yolo_names_path = "models/yolo_model/coco.names"
    net, classes = load_yolo_model(yolo_cfg_path, yolo_weights_path, yolo_names_path)

    image_path="/mnt/data/"
    image = cv2.imread(image_path)

    detections = detect_objects(image, net, classes)

    # Draw bounding boxes
    for (box, confidence, class_name) in detections:
        (x, y, w, h) = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{class_name}: {confidence:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image with detections
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
