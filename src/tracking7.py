from collections import defaultdict

import cv2
import random

import numpy
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
from typing import NoReturn, Dict, List

PENDING_THRESHOLD = 0.2
STABILITY_FRAMES = 5

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
        self.pending_embeddings = defaultdict(lambda: {"count": 0, "embs": []})
        self.next_temp_id = 0

    def track_people(self, known_embeddings: Dict[int, List[np.ndarray]]) -> Optional[np.ndarray]:
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
            current_embedding /= np.linalg.norm(current_embedding)
            box = boxes[i]

            # 2. Comparer avec tous les embeddings connus
            matched_id = None
            best_score = 1.0  # Cosine distance: plus petit = plus proche

            for known_id, emb_list in known_embeddings.items():
                for ref_emb in emb_list:
                    score = cosine(current_embedding, ref_emb)
                    if score < 0.19:  # Seuil à ajuster      quelque chose a jouer avec ce seuil
                        # print("score :",score)
                        if score < best_score:
                            best_score = score
                            matched_id = known_id
            label = None
            if matched_id is not None:
                label = f"ID {matched_id}"
                if len(known_embeddings[matched_id]) > 30:
                    known_embeddings[matched_id].pop(0)

                # smoothing pour éviter le bruit
                smoothed_emb = (
                        0.7 * known_embeddings[matched_id][-1] + 0.3 * current_embedding
                )
                smoothed_emb = smoothed_emb / np.linalg.norm(smoothed_emb)
                known_embeddings[matched_id].append(smoothed_emb)
            else:
                # 2. Vérif dans les pending IDs
                pending_match = None
                best_pending_score = 1.0
                for temp_id, data in self.pending_embeddings.items():
                    for ref_emb in data["embs"]:
                        score = cosine(current_embedding, ref_emb)
                        if score < PENDING_THRESHOLD and score < best_pending_score:
                            best_pending_score = score
                            pending_match = temp_id

                if pending_match:
                    # Mise à jour du pending existant
                    self.pending_embeddings[pending_match]["count"] += 1
                    self.pending_embeddings[pending_match]["embs"].append(current_embedding)

                    if self.pending_embeddings[pending_match]["count"] >= STABILITY_FRAMES:
                        # Validation → devient un vrai ID
                        new_id = generate_new_id()
                        known_embeddings[new_id] = self.pending_embeddings[pending_match]["embs"]
                        del self.pending_embeddings[pending_match]
                        matched_id = new_id
                        label = f"ID {new_id}"
                    else:
                        label = "Pending..."
                else:
                    # Nouveau pending
                    temp_id = f"temp_{self.next_temp_id}"
                    self.next_temp_id += 1
                    self.pending_embeddings[temp_id] = {"count": 1, "embs": [current_embedding]}
                    label = "Pending..."

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
