import os
import cv2
import random
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
from time import sleep
from typing import List
from typing import NoReturn

def correct_box(box: Boxes) -> bool:
    """
    Vérifie si la boîte donné en paramètre est correcte et apte à être utilisé.

    Args:
        box (Boxes): _description_

    Returns:
        NoReturn: _description_
    """
    return box.xyxy[0][2] < 1230 and box.xyxy[0][0] > 50 and box.conf[0] > 0.7


video_path = "angle1.mp4"
cap = cv2.VideoCapture(video_path)
model = YOLO("yolov8n.pt")

tracker = DeepSort(embedder="mobilenet", embedder_gpu=True, max_age=5, n_init=2)  # Configure as needed
colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(2000)]

known_embeddings = {}

frame_id = 0

next_id = 1

while True:
    #sleep(0.2)
    frame_id += 1
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0])[0]  # YOLOv8 returns a list; take the first element
    detections = []

    boxes = list(filter(correct_box, results.boxes))

    for box in boxes:
        #print("xyxy :", box.xyxy[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        # Format: ([left, top, width, height], confidence, class_id)
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))


    embeds = tracker.generate_embeds(raw_dets=detections, frame=frame)    
    
    print("Nombre d'embeds :", len(embeds))
    for i in range(len(embeds)):
        current_embedding = embeds[i]
        box = boxes[i]
        # 2. Comparer avec tous les embeddings connus
        matched_id = None
        best_score = 1.0  # Cosine distance: plus petit = plus proche
        best_avg = 1
        for known_id, emb_list in known_embeddings.items():
            length_embed = len(emb_list)
            current_avg = 0
            distances = [cosine(current_embedding, ref_emb) for ref_emb in emb_list]
            distances.sort()
            current_avg = sum(distances[:5]) / 5
            
            for ref_emb in emb_list:
                score = cosine(current_embedding, ref_emb)
                print(score)
                current_avg += score / length_embed
            
            if current_avg < 0.225:
                if current_avg < best_avg:
                    matched_id = known_id
                    best_avg = current_avg

        #print("I find something !",matched_id)
        label = None
        if matched_id:
            label = f"ID {matched_id}"
            if len(known_embeddings[matched_id]) > 35:
                known_embeddings[matched_id].pop(0)
            
            known_embeddings[matched_id].append(current_embedding)
        else:
            label = f"New ID {next_id}"
            known_embeddings[next_id] = [current_embedding]
            next_id += 1
        # Affichage
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[matched_id if matched_id is not None else 0], 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_ITALIC, 0.7, colors[matched_id if matched_id is not None else 0], 2)

        #x1, y1, x2, y2 = map(int, track.to_ltrb())
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
