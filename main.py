import math
from ultralytics import YOLO
import cv2
import cvzone
# from sort import Sort
# cap=cv2.VideoCapture(0) =>Web cam
# cap.set(3,1280) => these two are size of the camera
# cap.set(4,720)
cap = cv2.VideoCapture('E:/yolo/traffic.mp4')
# D:\MachineLearning\Project1\SIH\main.py
model = YOLO('../yolo-weights/yolov8x-worldv2.pt')

className = ['ambulance', 'army vehicle', 'auto rickshaw', 'bicycle', 'bus', 'car', 'garbagevan', 'human hauler',
             'minibus', 'minivan', 'motorbike', 'pickup', 'policecar', 'rickshaw', 'scooter', 'suv', 'taxi',
             'three wheelers -CNG-', 'truck', 'van', 'wheelbarrow']

print(len(className))
mask = cv2.imread("Videos/Untitled design (1).png")
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
while True:
    success, img = cap.read()
    # imgRegion=cv2.bitwise_and(img,mask)
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = box.conf[0]
            conf = math.ceil(conf * 100) / 100
            cls = int(box.cls[0])
            # currclass=className[cls]

            cvzone.putTextRect(img, f' {conf}', (max(0, x1), max(0, y1)),
                               thickness=1, scale=0.6, offset=3)

    cv2.imshow("Image", img)
    # cv2.imshow("image",imgRegion)
    cv2.waitKey(1)