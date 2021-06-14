# Source  : https://github.com/deepraj1729/yplate/blob/master/yplate/commands.py

import cv2
import json
import numpy as np


def detect(img_name, cfg, weights, classes, hide_img=False, hide_out=False):
    try:
        # Load image file
        img = cv2.imread(img_name)

        # Load Model
        net = cv2.dnn.readNet(weights, cfg)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # image dimensions
        height, width, channels = img.shape

        # Dynamic Detected plate Font Text
        if width > height:
            text_font_size = int((width / 1000) * 5)
        else:
            text_font_size = int((height / 1000) * 5)
        colors = (0, 255, 34)

        # Create the pipeline
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Running YOLO algorithm
        net.setInput(blob)
        outs = net.forward(output_layers)

        cropped_plate = []
        crop_rect = []
        class_ids = []
        confidences = []
        boxes = []

        # Loop to predict plates
        for out in outs:
            for detection in out:

                # Detect confidence
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Detected plates put inside the image
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Detected rectangle
                crop_rect.append([x, y, w, h])

                # Cropped detected rectangle
                crop = img[y:y + h, x:x + w]
                cropped_plate.append(crop)
                cv2.putText(img, label, (x, y - 20), font, text_font_size, (255, 255, 80), 2)

        no_of_detected_plates = len(crop_rect)
        confidences = confidences[-no_of_detected_plates:]

        # Show image (default show image)
        if not hide_img:
            cv2.imshow("Output Image", img)
            key = cv2.waitKey(2000) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()

        # Silence output or not
        if not hide_out:
            if no_of_detected_plates > 0:
                data = {
                    "output": {
                        "model": "YOLO v3",
                        "confidence": "{}".format(confidences),
                        "cropped_rectangles": "{}".format(crop_rect),
                        "plates_found": "{}".format(no_of_detected_plates),
                        "class_ids": "{}".format(class_ids)
                    }
                }
                print(json.dumps(data, indent=2))
                print("\nPlat nomor terdeteksi !!")
            else:
                print("Plat nomor tidak terdeteksi ")

        return confidences, crop_rect, class_ids
    except AttributeError as error:
        print("File not found in the given path '{}' ".format(img_name))

    except Exception as e:
        print(f"{e} \n Oops!Couldn't detect plate due to some errors. Check command line inputs")
