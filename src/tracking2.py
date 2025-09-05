import os
import cv2
import random
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

video_path = "video-couloire-720.mp4"
cap = cv2.VideoCapture(video_path)
model = YOLO("yolov8l.pt")

tracker = DeepSort(embedder="mobilenet", embedder_gpu=True, max_age=5, n_init=2)  # Configure as needed
colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(100)]

person_data = {}

frame_id = 0
while True:

    frame_id += 1
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]  # YOLOv8 returns a list; take the first element
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        # Format: ([left, top, width, height], confidence, class_id)
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

    #tracks = tracker.update_tracks(detections, frame=frame)

    #for track in tracks:
    #   
    #  if not track.is_confirmed():
    #        continue
    #    track_id = track.track_id
    #    x1, y1, x2, y2 = map(int, track.to_ltrb())
    #    color = colors[int(track_id) % len(colors)]
    #    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    #    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if len(person_data) > 3:
        tracks = tracker.update_tracks(detections, frame=frame, embeds=person_data)
    else:
        tracks = tracker.update_tracks(detections, frame=frame)
    
    print(person_data)
    for track in tracks:
        print(track.features)

        if not track.is_confirmed():
            continue
        
        track_id = str(track.track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        height = y2 - y1
        width = x2 - x1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        if track_id not in person_data:
            person_data[track_id] = {
                "track_id": track_id,
                "frames_seen": [],
                "positions": [],
                "heights": [],
                "embeddings": []
            }

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        color = colors[int(track_id) % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        person_data[track_id]["frames_seen"].append(frame_id)
        person_data[track_id]["positions"].append([x1, y1, x2, y2])
        person_data[track_id]["heights"].append(height)

        if hasattr(track, "features"):
            person_data[track_id]["embeddings"].append(track.features)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
