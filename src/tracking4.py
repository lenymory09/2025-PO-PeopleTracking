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


def correct_box(box: Boxes) -> NoReturn:
    """
    Vérifie si la boîte donné en paramètre est correcte et apte à être utilisé.

    Args:
        box (Boxes): _description_

    Returns:
        NoReturn: _description_
    """
    return box.xyxy[0][2] < 1230 and box.xyxy[0][0] > 50 and box.conf[0] > 0.7


video_path = "video-couloire-720.mp4"
cap = cv2.VideoCapture(video_path)
model = YOLO("yolov8n.pt")

tracker = DeepSort(embedder="mobilenet", embedder_gpu=True, max_age=5, n_init=2)  # Configure as needed
colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(100)]

known_embeddings = {}

frame_id = 0

while True:
    #sleep(0.2)
    frame_id += 1
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]  # YOLOv8 returns a list; take the first element
    detections = []

    boxes = filter(correct_box, results.boxes)

    for box in boxes:
        #print("xyxy :", box.xyxy[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        # Format: ([left, top, width, height], confidence, class_id)
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

    tracks = tracker.update_tracks(detections, frame=frame)

    embeds = tracker.generate_embeds(raw_dets=detections, frame=frame)
    
    #print(embeds)
    #print("Knowns embeddings :", known_embeddings)
    print("Nombre d'embeds :", len(embeds))
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = str(track.track_id)

        if hasattr(track, "features") and len(track.features) > 0:
            current_embedding = track.features[-1]
        else:
            continue
        # sleep(0.5)
        if track_id not in known_embeddings:
            known_embeddings[track_id] = []

        known_embeddings[track_id].append(current_embedding)

        # 2. Comparer avec tous les embeddings connus
        matched_id = None
        best_score = 1.0  # Cosine distance: plus petit = plus proche

        for known_id, emb_list in known_embeddings.items():
            for ref_emb in emb_list:
                score = cosine(current_embedding, ref_emb)
                if score < 0.4:  # Seuil à ajuster      quelque chose a jouer avec ce seuil 
                    #print("score :",score)
                    if score < best_score:
                        best_score = score
                        matched_id = known_id
        #print("I find something !",matched_id)
        label = None
        if matched_id:
            label = f"Re-ID {matched_id}"
        else:
            label = f"New ID {track_id}"

        # Affichage
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_ITALIC, 0.7, (0,255,0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
