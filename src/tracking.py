import os
import cv2
from ultralytics import YOLO
import random
from deep_sort.tracker import Tracker

video_path = os.path.join('.', 'video.mp4')

cap = cv2.VideoCapture(video_path)

model = YOLO("yolov8n.pt")

ret, frame = cap.read()

tracker = Tracker(metric=1)

colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for j in range(10)]
while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data:
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            detections.append([x1, y1, x2, y2, score])
    
        tracker.update(detections)
    
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)            

    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()