import cv2
import random
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
from typing import NoReturn, Dict, List
import torchreid
import torch
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((256, 128)),  # format standard ReID
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

DISTANCE_THRESHOLD = 0.25

# Charger un modèle OSNet pré-entraîné
model_torch = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    loss='softmax',
    pretrained=True
)

model_torch.eval()  # mode inference


def extract_embeddings(img, boxes, model):
    embeds = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        person_crop = img[y1:y2, x1:x2]  # crop OpenCV
        person_crop = Image.fromarray(person_crop[:, :, ::-1])  # BGR->RGB
        tensor = transform(person_crop).unsqueeze(0)

        with torch.no_grad():
            emb = model_torch(tensor)
        embeds.append(emb.squeeze().numpy())
    return embeds


def is_correct_box(box: Boxes) -> bool:
    """
    Vérifie si la boîte donné en paramètre est correcte et apte à être utilisé.

    Args:
        box (Boxes): _description_

0        NoReturn: _description_
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
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.next_temp_id = 0

    def track_people(self, known_embeddings: Dict[int, List[np.ndarray]], assigned_ids) -> NoReturn:
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

        # embeds = self.generate_embeds(raw_dets=detections, frame=frame)

        embeds = extract_embeddings(frame, [box.xyxy[0] for box in boxes], model)

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
                if len(known_embeddings[matched_id]) > 30:
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
