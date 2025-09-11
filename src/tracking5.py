import cv2
import random
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
from typing import NoReturn, Dict, List

DISTANCE_THRESHOLD = 0.18
MAX_DESCRIPTION_NUMBER = 50

# def boxes_overlap(boxA, boxB) -> bool:
#     """
#     Vérifie si deux boîtes se chevauchents
#     :param boxA:
#     :param boxB:
#     :return:
#     """
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     return (xB - xA) > 0 and (yB - yA) > 0


def is_diverse(new_emb, emb_list, min_dist=0.1):
    return all(cosine(new_emb, e) > min_dist for e in emb_list)


def is_correct_box(box: Boxes, width: int) -> bool:
    """
    Vérifie si la boîte donné en paramètre est correcte et apte à être utilisé.

    Args:
        box (Boxes): _description_

0        NoReturn: _description_
    """
    x1, y1, x2, y2 = box.xyxy[0]
    width_box = x2 - x1
    height_box = y2 - y1
    is_dimensions_correct = width_box > 100 and height_box > 120
    return bool(x1 > 50 and x2 < width - 50 and box.conf[0] > 0.7) and is_dimensions_correct


def generate_new_id() -> int:
    global next_id
    next_id += 1
    return next_id


next_id = 1
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(2000)]
model = YOLO("yolov8l.pt")


class Camera(DeepSort):
    def __init__(self, source: str):
        super().__init__(embedder="mobilenet", embedder_gpu=True, max_age=5, n_init=2)
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.next_temp_id = 0

    def track_people(self, known_embeddings: Dict[int, List[np.ndarray]]) -> NoReturn:
        ret, frame = self.cap.read()
        height, width, _ = frame.shape
        print("Shape de la frame:", frame.shape)
        if not ret:
            return None

        results = model(frame, classes=[0])[0]  # YOLOv8 returns a list; take the first element
        detections = []
        boxes = list(filter(lambda box: is_correct_box(box, width), results.boxes))

        # Vérifier le chevauchement avec d'autres boîtes déjà traitées
        # result = []
        # for i, box in enumerate(boxes):
        #     x1, y1, x2, y2 = map(int, box.xyxy[0])
        #     current_box = (x1, y1, x2, y2)
        #     for j in range(i):
        #         prev_box = boxes[j]
        #         px1, py1, px2, py2 = map(int, prev_box.xyxy[0])
        #         prev_box_coords = (px1, py1, px2, py2)
        #
        #         if boxes_overlap(current_box, prev_box_coords):
        #             result.append(box)
        #
        #
        # for box in result:
        #     boxes.remove(box)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            # Format: ([left, top, width, height], confidence, class_id)
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        embeds = self.generate_embeds(raw_dets=detections, frame=frame)

        assigned_ids = []

        for i in range(len(embeds)):
            current_embedding = embeds[i]
            box = boxes[i]

            # 2. Comparer avec tous les embeddings connus
            matched_id = None
            best_score = 1.0  # Cosine distance: plus petit = plus proche

            for known_id, emb_list in known_embeddings.items():
                if known_id not in assigned_ids:
                    for ref_emb in emb_list:
                        score = cosine(current_embedding, ref_emb)
                        if score < DISTANCE_THRESHOLD and score < best_score:  # Seuil à ajuster
                            best_score = score
                            matched_id = known_id
            label = None
            if matched_id:
                label = f"ID {matched_id}"
                # if is_diverse(current_embedding, known_embeddings[matched_id]):
                if len(known_embeddings[matched_id]) > MAX_DESCRIPTION_NUMBER:
                    known_embeddings[matched_id].pop(0)

                known_embeddings[matched_id].append(current_embedding)

                assigned_ids.append(matched_id)
            else:
                new_id = generate_new_id()
                label = f"New ID {new_id}"
                assigned_ids.append(new_id)
                known_embeddings[new_id] = [current_embedding]

            # Affichage
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[matched_id if matched_id is not None else 0], 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_ITALIC, 0.7,
                        colors[matched_id if matched_id is not None else 0], 2)

        # x1, y1, x2, y2 = map(int, track.to_ltrb())

        return frame


class PersonTracker(object):
    known_embeddings: Dict[int, List[np.ndarray]] = {}
    cameras: List[Camera] = []

    def __init__(self, sources: List[str]):
        self.sources = sources
        for source in sources:
            self.cameras.append(Camera(source))

    def release(self):
        for camera in self.cameras:
            camera.cap.release()

    def get_frames(self):
        frames: List[np.ndarray] = []

        for camera in self.cameras:
            frames.append(camera.track_people(self.known_embeddings))

        return frames


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
