import cv2
import random

import numpy
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
from typing import NoReturn, Dict


def is_correct_box(box: Boxes) -> bool:
    """
    Vérifie si la boîte donné en paramètre est correcte et apte à être utilisé.

    Args:
        box (Boxes): _description_

    Returns:
        NoReturn: _description_
    """
    return bool(box.xyxy[0][2] < 1230 and box.xyxy[0][0] > 50 and box.conf[0] > 0.7)




def generate_new_id() -> int:
    global next_id
    next_id += 1
    return next_id

next_id = 1
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(2000)]
model = YOLO("yolov8n.pt")

class PersonTracker(DeepSort):
    def __init__(self, source: str):
        super().__init__(embedder="mobilenet", embedder_gpu=True, max_age=5, n_init=2)
        self.cap = cv2.VideoCapture(source)

    def track_people(self, known_embeddings: Dict[int, numpy.ndarray]) -> NoReturn:
        ret, frame = self.cap.read()
        if not ret:
            return None

        results = model(frame, classes=[0])[0]  # YOLOv8 returns a list; take the first element
        detections = []

        boxes = list(filter(is_correct_box, results.boxes))

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            # Format: ([left, top, width, height], confidence, class_id)
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        embeds = self.generate_embeds(raw_dets=detections, frame=frame)

        print("Nombre d'embeds :", len(embeds))
        for i in range(len(embeds)):
            current_embedding = embeds[i]
            box = boxes[i]
            # 2. Comparer avec tous les embeddings connus
            matched_id = None
            best_score = 1.0  # Cosine distance: plus petit = plus proche

            for known_id, emb_list in known_embeddings.items():
                for ref_emb in emb_list:
                    score = cosine(current_embedding, ref_emb)
                    if score < 0.2:  # Seuil à ajuster      quelque chose a jouer avec ce seuil
                        # print("score :",score)
                        if score < best_score:
                            best_score = score
                            matched_id = known_id
            label = None
            if matched_id:
                label = f"ID {matched_id}"
                if len(known_embeddings[matched_id]) > 45:
                    known_embeddings[matched_id].pop(0)

                known_embeddings[matched_id].append(current_embedding)
            else:
                new_id = generate_new_id()
                label = f"New ID {new_id}"
                known_embeddings[new_id] = [current_embedding]
            # Affichage
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[matched_id if matched_id is not None else 0], 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_ITALIC, 0.7,
                        colors[matched_id if matched_id is not None else 0], 2)

            # x1, y1, x2, y2 = map(int, track.to_ltrb())

        return frame


if __name__ == '__main__':
    tracker = PersonTracker("angle2.mp4")

    known_embeddings = {}
    while True:
        frame = tracker.track_people(known_embeddings)
        cv2.imshow("source", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.cap.release()
    cv2.destroyAllWindows()
