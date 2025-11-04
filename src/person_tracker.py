import cv2
import random
import numpy as np
import torch
import torchreid
from torchvision import transforms
from ultralytics.engine.results import Boxes
from scipy.spatial.distance import cosine
from typing import List, Tuple
import os
from PIL import Image
from utils import euclidean_distance
import threading

torch.set_num_threads(os.cpu_count())

DISTANCE_THRESHOLD = 0.36  # Tighter threshold
MAX_DESCRIPTION_NUMBER = 200  # More manageable number
MIN_TRACK_LENGTH = 5  # Require minimum track length

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(2000)]


# Use multiple reference embeddings for comparison
# scores = []
# for ref_emb in emb_list[-20:]:  # Use recent embeddings
# Try multiple distance metrics
# cosine_dist = cosine(embed, ref_emb)
# euclidean_dist = euclidean_distance(embed, ref_emb)

# Combined score
# combined_score = 0.7 * cosine_dist + 0.3 * (euclidean_dist / 10.0)
# scores.append(combined_score)

# Use best match from recent embeddings
# if scores:
# current_best = min(scores)
# if current_best < DISTANCE_THRESHOLD and current_best < best_score:
# best_score = current_best
# matched_id = known_id

# return matched_id


def filter_boxes_by_dimensions(boxes: List[Boxes], width: int) -> List[Boxes]:
    return list(filter(lambda box: is_correct_box(box, width), boxes))


def boxes_overlap(box1: List[int], box2) -> bool:
    """
    Vérifie si deux boîtes se chevauchents
    :param box1: boite 1 à analyser
    :param box2: boite 2 à analyser
    :return:
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    return (xB - xA) > 0 and (yB - yA) > 0


def filter_by_aspect_ratio(boxes: List[Boxes], min_ratio: float = 0.1, max_ratio: float = 0.8) -> List[Boxes]:
    """Filter boxes by human-like aspect ratios"""
    filtered = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        width = x2 - x1
        height = y2 - y1
        if height == 0:
            continue
        aspect_ratio = width / height
        if min_ratio <= aspect_ratio <= max_ratio:
            filtered.append(box)
    return filtered


def filter_boxes_by_overlapping(boxes: List[Boxes]) -> List[Boxes]:
    boxes = sorted(boxes, key=lambda box: box.xyxy[0][0])
    filtered_boxes = []
    for i in range(len(boxes)):
        box1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            box2 = boxes[j]
            if boxes_overlap(box1.xyxy[0], box2.xyxy[0]):
                best_box = max(box1, box2, key=lambda box: box.xyxy[0][3] - box.xyxy[0][1])
                if best_box not in filtered_boxes:
                    filtered_boxes.append(best_box)

    return filtered_boxes


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
    return bool(x1 > 50 and x2 < width - 50 and box.conf[0] > 0.75) and is_inside_frame  # and is_dimensions_correct


class EnhancedPersonTracker:
    def __init__(self, config):
        self.tracked_persons = {}
        self.track_history = {}  # Track movement patterns
        self.appearance_history = {}  # Store multiple appearances
        self.next_id = 0
        self.color_map = {}
        self.max_gallery_size = config['reid']['max_gallery_size']
        self.temporal_frames = config['reid']['temporal_frames']
        self.threshold_reid = config['reid']['threshold']
        self.device = torch.device(config['reid']['device'])

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['reid']['norm_mean'],
                                 std=config['reid']['norm_std'])
        ])

        model_name = config['models']['reid_gpu'] if config['reid']['device'] in ["cuda", "mps"] else config['models'][
            'reid_cpu']

        self.reid_model = torchreid.models.build_model(
            name=model_name,
            num_classes=1501,
            loss="softmax",
            pretrained=True,
            use_gpu=torch.backends.mps.is_available()
        )

        if model_name == "osnet_ain_x1_0":
            torchreid.utils.load_pretrained_weights(self.reid_model, 'models/osnet_ain_market1501_60.pth')

        self.reid_model.eval()

        self.lock = threading.Lock()

        # self.feature_extractor = FeatureExtractor("resnet50", "models/resnet50_market.pth.tar-60", verbose=False, device="cpu")

    def update_gallery(self, pid, feat):
        with self.lock:
            entry = self.tracked_persons[pid]
            features = entry['features']
            entry['features'] = np.vstack((features[-(self.max_gallery_size - 1):], feat))
            entry['temp_frames'] = min(entry['temp_frames'] + 1, self.temporal_frames)

    def _create_id(self, feat):
        # if isinstance(feat, torch.Tensor):
        #    feat = feat.detach().cpu().numpy().squeeze()
        with self.lock:
            pid = self.next_id
            self.next_id += 1
            color = tuple(np.random.randint(0, 255, 3).tolist())
            self.color_map[pid] = color
            self.tracked_persons[pid] = {
                'features': np.array([feat]),
                'temp_frames': 0,
                'name': None,
                'color': color
            }
            print(f"Created ID {pid} with color {color}")
            return pid

    def match_person(self, embed: np.ndarray, assigned_ids: List[int]):
        matched_id = None
        best_score = float('inf')
        # print(self.tracked_persons)
        with self.lock:
            for known_id, person in self.tracked_persons.items():
                person_embed = person['features'].mean(0)
                if known_id in assigned_ids:
                    continue
                # Use multiple reference embeddings for comparison
                # scores = []
                # for ref_emb in emb_list[-20:]:  # Use recent embeddings
                # Try multiple distance metrics
                cosine_dist = cosine(embed, person_embed)
                euclidean_dist = euclidean_distance(embed, person_embed)

                # Combined score
                score = 0.7 * float(cosine_dist) + 0.3 * (euclidean_dist / 10.0)

                # Use best match from recent embeddings
                if score < self.threshold_reid and score < best_score:
                    best_score = score
                    matched_id = known_id

        if matched_id is None:
            matched_id = self._create_id(embed)
        else:
            self.update_gallery(matched_id, embed)

        assigned_ids.append(matched_id)

        return matched_id

    def generate_embeddings(self, frame: np.ndarray, boxes):
        result = []
        for box in boxes:
            result.append(self.extract_embedding(frame, box))

        return result

    def extract_embedding(self, img: np.ndarray, box: Tuple):
        """Improved embedding extraction with preprocessing"""
        x1, y1, x2, y2 = map(int, box)
        # x1, y1, width, height = map(int, box)
        # x2, y2 = x1 + width, y1 + height

        # Expand box slightly for better context
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        y2 = min(img.shape[0], y2 + padding)

        person_crop = img[y1:y2, x1:x2]

        # Skip if crop is too small
        if person_crop.size == 0 or person_crop.shape[0] < 50 or person_crop.shape[1] < 25:
            return None

        # Enhanced preprocessing
        person_crop = Image.fromarray(person_crop[:, :, ::-1])

        # Apply multiple augmentations and average embeddings
        embeddings = []
        # for augment in [False, True]:
        #     if augment:
        #         tensor = self.transform(transforms.functional.hflip(person_crop)).unsqueeze(0)
        #     else:
        #         tensor = self.transform(person_crop).unsqueeze(0)
        #
        #     with torch.no_grad():
        #         emb = self.reid_model(tensor)
        #         embeddings.append(emb.squeeze().numpy())
        tensor = self.transform(person_crop).unsqueeze(0)
        with torch.no_grad():
            emb = self.reid_model(tensor)
            embeddings.append(emb.squeeze().numpy())

        # Average the embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding / np.linalg.norm(avg_embedding)  # L2 normalize

    def extract_features(self, bgr_imgs) -> np.ndarray:
        batch = []
        for img in bgr_imgs:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_t = self.transform(img_pil).unsqueeze(0)
            batch.append(img_t)

        with torch.no_grad():
            batch_t = torch.cat(batch).to(self.device)
            feats = self.reid_model(batch_t).cpu().numpy()

        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
        return feats / norms

    def calc_nb_persons(self):
        return len(self.tracked_persons)

    def generate_label(self, pid: int) -> str:
        """
        Generate the label for the person Box
        :param pid: Person ID
        :return: the label generated
        """
        nb_embeddings = len(self.tracked_persons[pid]['features'])
        label = f"ID {pid} : {'Max' if nb_embeddings == self.max_gallery_size else nb_embeddings}"
        return label
