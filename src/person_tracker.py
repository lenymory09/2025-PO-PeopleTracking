import cv2
import random
import numpy as np
import torch
import torchreid
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional, NoReturn, Tuple
from utils import chrono

DISTANCE_THRESHOLD = 0.25
MAX_DESCRIPTION_NUMBER = 400

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model_osnet = torchreid.models.build_model(
    name="squeezenet1_1",
    num_classes=1000,
    loss="softmax",
    pretrained=True
)

model_osnet.eval()
model_deepsort = DeepSort(embedder="mobilenet", embedder_gpu=True, max_age=5, n_init=2)


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
        :param box: Boite à analyser
        :param width: largeur de la frame analysé


    Returns:
        :returns: vrai si la boite est juste et faux sinon.
    """
    x1, y1, x2, y2 = box.xyxy[0]
    # width_box = x2 - x1
    # height_box = y2 - y1
    # is_dimensions_correct = width_box > 100 and height_box > 120
    is_inside_frame = (
            0 <= x1 < width and
            0 <= x2 <= width
    )
    return bool(x1 > 50 and x2 < width - 50 and box.conf[0] > 0.7) and is_inside_frame  # and is_dimensions_correct


def generate_new_id() -> int:
    """
    Génere le nouvelle ID et le retourne
    :return: le nouvelle ID
    """
    global next_id
    next_id += 1
    return next_id


def extract_embedding(img: np.ndarray, box, model_ia):
    x1, y1, width, height = map(int, box[0])
    x2, y2 = x1 + width, y1 + height
    person_crop = img[y1:y2, x1:x2]
    person_crop = Image.fromarray(person_crop[:, :, ::-1])
    tensor = transform(person_crop).unsqueeze(0)

    with torch.no_grad():
        emb = model_ia(tensor)
    return emb.squeeze().numpy()


def generate_embeddings(img, boxes, ai):
    if ai == "osnet":
        result = []
        for box in boxes:
            result.append(extract_embedding(img, box, model_osnet))

        return result

    return model_deepsort.generate_embeds(img, raw_dets=boxes)


def draw_rectangle(f: np.ndarray, box: Boxes, label: str, person_id: int) -> np.ndarray:
    """
    Draw the rectangle and the ids in the image.
    :param f: Frame to draw onto.
    :param box: Box of the person
    :param label: Label of the person (ID and nb embeddings)
    :param person_id:
    :return:
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(f, (x1, y1), (x2, y2), colors[person_id if person_id is not None else 0], 2)
    cv2.putText(f, label, (x1, y1 - 10), cv2.FONT_ITALIC, 0.7,
                colors[person_id if person_id is not None else 0], 2)

    return f


def generate_detections(frame: np.ndarray, frame_width: int) -> Tuple[List[Tuple[List, float, int]], List[Boxes]]:
    """
    Generate the detections for the ReID Algorithm.
    :param frame: Frame to analyse
    :param frame_width: Width of the frame.
    :return: The detections and the boxes
    """
    results = model(frame, classes=[0], device="cpu")[0]
    detections = []
    boxes = list(filter(lambda detection: is_correct_box(detection, frame_width), results.boxes))
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        # Format: ([left, top, width, height], confidence, class_id)
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))
    return detections, boxes


def process_id(embed: np.ndarray, matched_id: Optional[int], known_persons: Dict[int, List[np.ndarray]],
               assigned_ids: List[int]) -> str:
    """
    Save the id in the known_person or create it and add it to the assigned_ids
    :param embed: person description
    :param matched_id: ID matched with the person (or None)
    :param known_persons: Known person dictionary
    :param assigned_ids: List of assigned IDs
    :return: the label of the box
    """

    if matched_id:
        # if is_diverse(current_embedding, known_embeddings[matched_id]):
        known_persons[matched_id].append(embed)

        if len(known_persons[matched_id]) > MAX_DESCRIPTION_NUMBER:
            known_persons[matched_id].pop(0)

        nb_embeddings = len(known_persons[matched_id])

        label = f"ID {matched_id} : {'Max' if nb_embeddings == MAX_DESCRIPTION_NUMBER else nb_embeddings}"
    else:
        matched_id = generate_new_id()
        label = f"New ID {matched_id}"
        known_persons[matched_id] = [embed]

    assigned_ids.append(matched_id)
    return label


def get_nearest_person(embed: np.ndarray, known_persons: Dict[int, List[np.ndarray]], assigned_ids: List[int]):
    matched_id = None
    best_score = 1.0
    for known_id, emb_list in known_persons.items():
        if known_id not in assigned_ids:
            for ref_emb in emb_list:
                score = cosine(embed, ref_emb)
                if score < DISTANCE_THRESHOLD and score < best_score:  # Seuil à ajuster
                    best_score = score
                    matched_id = known_id

    return matched_id


next_id = 0
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(2000)]
model = YOLO("yolo11n.pt")


class Camera:
    """
    Represents one camera of the PersonTracker
    """

    def __init__(self, source: str, ai: str):
        self.model = ai
        self.source = source
        self.cap = cv2.VideoCapture(source)
        assert self.cap.isOpened(), "Cap is not opened."

    def read(self):
        return self.cap.read()

    @chrono
    def track_people(self, known_persons: Dict[int, List[np.ndarray]]) -> Optional[np.ndarray]:
        """
        Read the stream video and return the analysed frame.
        :param known_persons: Known person by the program.
        :return: The analysed frame.
        """
        ret, frame = self.read()
        height, width, _ = frame.shape
        print("Shape :", frame.shape)
        if not ret:
            return None
        if cv2.waitKey(1) == ord('q'):
            raise KeyboardInterrupt()

        detections, boxes = generate_detections(frame, width)
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

        embeds = generate_embeddings(frame, detections, self.model)

        assigned_ids = []

        for current_embedding, box in zip(embeds, boxes):
            # Comparaison avec les embeddings connus.
            matched_id = get_nearest_person(current_embedding, known_persons, assigned_ids)

            label = process_id(current_embedding, matched_id, known_persons, assigned_ids)

            # Affichage
            frame = draw_rectangle(frame, box, label, matched_id)

        return frame


class PersonTracker:
    known_embeddings: Dict[int, List[np.ndarray]] = {}
    cameras: List[Camera] = []

    def __init__(self, sources: List[str], ai: str):
        self.model: str = ai
        if ai == 'deepsort':
            self.model_ai = DeepSort(embedder="mobilenet", embedder_gpu=True, max_age=5, n_init=2)
        self.sources = sources
        for source in sources:
            self.cameras.append(Camera(source, ai))

    def release(self):
        for camera in self.cameras:
            camera.cap.release()

    def get_frames(self):
        frames: List[np.ndarray] = []

        for camera in self.cameras:
            frames.append(camera.track_people(self.known_embeddings))

        return frames

    def start(self) -> NoReturn:
        while True:
            for i, camera in enumerate(self.cameras, start=1):
                frame_analysed = camera.track_people(self.known_embeddings)
                cv2.imshow(f"stream {i}", frame_analysed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt("Fin du programme")
