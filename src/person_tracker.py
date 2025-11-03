import queue
import time
from collections import deque
import cv2
import random
import numpy as np
import torch
import torchreid
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional, Tuple
import os
from PIL import Image
from utils import extract_image_patch, euclidean_distance, draw_person_box
from queue import Queue
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
    return bool(x1 > 50 and x2 < width - 50 and box.conf[0] > 0.7) and is_inside_frame  # and is_dimensions_correct


class EnhancedPersonTracker:
    def __init__(self, config):
        self.tracked_persons = {}
        self.track_history = {}  # Track movement patterns
        self.appearance_history = {}  # Store multiple appearances
        self.sources: List[str] = config['video']['sources']
        self.cameras: List[Camera] = []
        self.next_id = 0
        self.color_map = {}
        self.max_gallery_size = config['reid']['max_gallery_size']
        self.temporal_frames = config['reid']['temporal_frames']
        self.threshold_reid = config['reid']['threshold']
        for idx, source in enumerate(self.sources):
            self.cameras.append(Camera(source, self, config, idx))
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
        entry = self.tracked_persons[pid]
        features = entry['features']
        entry['features'] = np.vstack((features[-(self.max_gallery_size - 1):], feat))
        entry['temp_frames'] = min(entry['temp_frames'] + 1, self.temporal_frames)

    def create_id(self, feat):
        # if isinstance(feat, torch.Tensor):
        #    feat = feat.detach().cpu().numpy().squeeze()

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

    def update_track_history(self, track_id: int, box: Boxes, frame_idx: int, embedding: np.ndarray = None):
        """
        Met à jour l'historique d'un track avec informations temporelles et spatiales
        """
        if track_id not in self.track_history:
            # Initialisation du track
            self.track_history[track_id] = {
                'first_seen': frame_idx,
                'last_seen': frame_idx,
                'positions': deque(maxlen=100),  # Dernières positions
                'appearances': 0,  # Nombre total d'apparitions
                'confidence_scores': deque(maxlen=50),
                'embedding_history': deque(maxlen=20),  # Derniers embeddings
                'active': True,
                'color': colors[track_id],
                'last_boxes': deque(maxlen=10)  # Dernières boîtes
            }

        # Mise à jour des informations
        history = self.track_history[track_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Ajout des nouvelles données
        history['positions'].append((center_x, center_y, frame_idx))
        history['confidence_scores'].append(float(box.conf[0]))
        history['last_boxes'].append(box.xyxy[0])
        history['last_seen'] = frame_idx
        history['appearances'] += 1

        if embedding is not None:
            history['embedding_history'].append(embedding)

        # Calcul des statistiques en temps réel
        self._calculate_track_stats(track_id)

    def match_person(self, embed: np.ndarray, assigned_ids: List[int]):
        matched_id = None
        best_score = float('inf')
        # print(self.tracked_persons)
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
            matched_id = self.create_id(embed)
        else:
            self.update_gallery(matched_id, embed)

        assigned_ids.append(matched_id)

        return matched_id

    def old_get_nearest_person(self, embed: np.ndarray, assigned_ids: List[int]):
        matched_id = None
        best_score = 1.0
        for known_id, emb_list in self.tracked_persons.items():
            if known_id not in assigned_ids:
                for ref_emb in emb_list:
                    score = cosine(embed, ref_emb)
                    # score = euclidean_distance(embed, ref_emb)
                    if score < DISTANCE_THRESHOLD and score < best_score:  # Seuil à ajuster
                        best_score = score
                        matched_id = known_id

        return matched_id

    def _calculate_track_stats(self, track_id: int):
        """
        Calcule les statistiques d'un track (vitesse, direction, etc.)
        """
        if track_id not in self.track_history:
            return

        history = self.track_history[track_id]
        positions = list(history['positions'])

        if len(positions) < 2:
            return

        # Calcul de la vitesse moyenne
        recent_positions = positions[-5:]  # 5 dernières positions
        speeds = []
        directions = []

        for i in range(1, len(recent_positions)):
            x1, y1, t1 = recent_positions[i - 1]
            x2, y2, t2 = recent_positions[i]

            # Vitesse en pixels par frame
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            time_diff = t2 - t1
            speed = distance / time_diff if time_diff > 0 else 0
            speeds.append(speed)

            # Direction
            if distance > 0:
                direction = np.arctan2(y2 - y1, x2 - x1)
                directions.append(direction)

        # Mise à jour des statistiques
        history['avg_speed'] = np.mean(speeds) if speeds else 0
        history['max_speed'] = np.max(speeds) if speeds else 0
        history['avg_direction'] = np.mean(directions) if directions else 0
        history['track_length'] = len(positions)

        # Calcul de la trajectoire prévue
        if len(positions) >= 3:
            self._predict_future_position(track_id)

    def _predict_future_position(self, track_id: int):
        """
        Prédit la position future basée sur la trajectoire
        """
        history = self.track_history[track_id]
        positions = list(history['positions'])[-5:]  # 5 dernières positions

        if len(positions) < 3:
            return

        # Régression linéaire simple pour prédiction
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        frames = [p[2] for p in positions]

        # Prédiction pour la frame suivante
        if len(set(frames)) > 1:
            try:
                # Prédiction X
                x_trend = np.polyfit(frames, x_coords, 1)
                next_x = np.polyval(x_trend, frames[-1] + 1)

                # Prédiction Y
                y_trend = np.polyfit(frames, y_coords, 1)
                next_y = np.polyval(y_trend, frames[-1] + 1)

                history['predicted_position'] = (next_x, next_y)
            except Exception as _:
                history['predicted_position'] = None

    def update_tracking_logic(self, embed: np.ndarray, box: Boxes, frame_idx: int):
        """Enhanced tracking with temporal consistency"""
        # Add temporal filtering
        matched_id = self.get_nearest_person_with_temporal(embed, box, frame_idx)

        if matched_id:
            self.update_track_history(matched_id, box, frame_idx)

        return matched_id

    def get_nearest_person_with_temporal(self, embed: np.ndarray, box: Boxes, frame_idx: int):
        """Matching that considers temporal and spatial context"""
        candidates = {}

        for known_id, emb_list in self.tracked_persons.items():
            # Check if this ID was recently seen
            if known_id in self.track_history:
                last_seen = self.track_history[known_id]['last_seen']
                frames_since_seen = frame_idx - last_seen

                # Adjust threshold based on time since last seen
                temporal_factor = max(0.1, 1.0 - (frames_since_seen * 0.01))
                adjusted_threshold = DISTANCE_THRESHOLD * temporal_factor
            else:
                adjusted_threshold = DISTANCE_THRESHOLD

            # Calculate similarity
            recent_embs = emb_list[-5:]  # Use most recent embeddings
            similarities = [1 - cosine(embed, ref_emb) for ref_emb in recent_embs]
            max_similarity = max(similarities) if similarities else 0

            if max_similarity > (1 - adjusted_threshold):
                candidates[known_id] = max_similarity

        if candidates:
            return max(candidates.items(), key=lambda x: x[1])[0]
        return None

    def get_track_info(self, track_id: int) -> Dict:
        """
        Retourne les informations d'un track spécifique
        """
        if track_id in self.track_history:
            return self.track_history[track_id]
        return {}

    def get_active_tracks(self, current_frame: int, max_frames_missing: int = 30) -> List[int]:
        """
        Retourne la liste des tracks actifs (vus récemment)
        """
        active_tracks = []

        for track_id, history in self.track_history.items():
            frames_missing = current_frame - history['last_seen']
            if frames_missing <= max_frames_missing and history['active']:
                active_tracks.append(track_id)

        return active_tracks

    def cleanup_old_tracks(self, current_frame: int, max_age_frames: int = 100):
        """
        Nettoie les tracks trop anciens
        :param max_age_frames:
        :param current_frame:
        :return:
        """
        tracks_to_remove = []

        for track_id, history in self.track_history.items():
            age = current_frame - history['last_seen']
            if age > max_age_frames:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.track_history[track_id]
            # Optionnel: supprimer aussi les embeddings
            if track_id in self.tracked_persons:
                del self.tracked_persons[track_id]

    def draw_track_history(self, frame: np.ndarray, track_id: int) -> np.ndarray:
        """
        Dessine l'historique du track sur la frame
        """
        if track_id not in self.track_history:
            return frame

        history = self.track_history[track_id]
        positions = list(history['positions'])

        # Dessiner la trajectoire
        for i in range(1, len(positions)):
            x1, y1, _ = positions[i - 1]
            x2, y2, _ = positions[i]

            # Couleur qui change avec le temps (plus récent = plus clair)
            alpha = i / len(positions)
            color = (
                int(colors[track_id][0] * alpha),
                int(colors[track_id][1] * alpha),
                int(colors[track_id][2] * alpha)
            )

            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Dessiner la position prédite
        if 'predicted_position' in history and history['predicted_position']:
            pred_x, pred_y = history['predicted_position']
            cv2.circle(frame, (int(pred_x), int(pred_y)), 8, (0, 255, 255), -1)
            cv2.putText(frame, "Fred", (int(pred_x), int(pred_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return frame

    def generate_embeddings(self, frame: np.ndarray, boxes):
        result = []
        for box in boxes:
            result.append(self.extract_embedding(frame, box))

        return result

    def extract_embedding(self, img: np.ndarray, box: Tuple):
        """Improved embedding extraction with preprocessing"""
        # x1, y1, x2, y2 = map(int, box)
        x1, y1, width, height = map(int, box[0])
        x2, y2 = x1 + width, y1 + height

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

    def extract_features(self, bgr_imgs):
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

class Camera:
    def __init__(self, source: str, reid: EnhancedPersonTracker, config, vid_idx):
        self.running = False
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.reid = reid
        self.frame_count = 0
        self.tracker = DeepSort(
            max_age=config['tracker']['max_age'],
            n_init=config['tracker']['n_init'],
            max_cosine_distance=config['tracker']['max_cosine_distance'],
            nn_budget=config['tracker']['nn_budget'],
        )
        self.detection_config = config['detection']
        self.track_id_to_pid: Dict[int, int] = {}
        self.yolo = YOLO(config['models']['yolo'])
        self.frame_queue: Optional[Queue] = None
        self.vid_idx = vid_idx

    def generate_detections(self, frame: np.ndarray) -> Tuple[List[Tuple], List[Boxes]]:
        """
        Generate the detections for the ReID Algorithm.
        :param frame: Frame to analyse
        :return: The detections and the boxes
        """
        _, frame_width, _ = frame.shape

        results = \
        self.yolo(frame, classes=[self.detection_config['person_class_id']], device=self.detection_config['device'],
                  conf=self.detection_config['confidence_threshold'], iou=0.5, verbose=False)[0]
        detections = []
        boxes = results.boxes
        # boxes = filter_boxes_by_dimensions(boxes, frame_width)
        # boxes = filter_boxes_by_overlag(boxes)
        # boxes = filter_by_aspect_ratio(boxes, 0.4)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            # Format: ([left, top, width, height], confidence, class_id)
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))
        return detections, boxes

    def process_id(self, embed: np.ndarray, matched_id: Optional[int],
                   assigned_ids: List[int]) -> int:
        """
        Save the id in the known_person or create it and add it to the assigned_ids
        :param embed: person description
        :param matched_id: ID matched with the person (or None)
        :param assigned_ids: List of assigned IDs
        :return: the label of the box
        """
        # self.reid.tracked_persons[matched_id].append(embed)

        # if len(self.tracker.tracked_persons[matched_id]) > MAX_DESCRIPTION_NUMBER:
        #    self.tracker.tracked_persons[matched_id].pop(0)

    def read(self):
        return self.cap.read()

    # @chrono
    def process_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.read()
        if not ret:
            return None

        height, width, _ = frame.shape
        self.frame_count += 1

        detections, boxes = self.generate_detections(frame)

        # tracks = self.tracker.update_tracks(detections, frame=frame)
        # bboxes = []
        # crops = []

        # good_tracks = []
        # for trk in tracks:
        #    if trk.is_confirmed() and trk.time_since_update <= 1:
        #        good_tracks.append(trk)
        #        l, t, r, b = map(int, trk.to_ltrb())
        #        if (r - l) > 10 and (b - t) > 10:
        #            bboxes.append((l, t, r, b))
        #            crops.append(frame[t:b, l:r])

        # if crops:
        # embeds = self.reid.extract_features(crops).tolist()
        embeds = self.reid.generate_embeddings(frame, detections)
        # embeds = list(map(lambda embed: embed / np.linalg.norm(embed), embeds))

        assigned_ids = []

        for current_embedding, box in zip(embeds, boxes):
            pid = self.reid.match_person(current_embedding, assigned_ids)
            label = self.reid.generate_label(pid)
            color = self.reid.tracked_persons[pid]['color']
            # Mise à jour de l'historique
            # if matched_id:
            # self.tracker.update_track_history(matched_id, box, self.frame_count, current_embedding)

            # Dessiner l'historique du track
            # frame = self.tracker.draw_track_history(frame, matched_id)

            draw_person_box(frame, box.xyxy[0], label, color)

        # Nettoyage périodique des anciens tracks
        if self.frame_count % 100 == 0:  # Toutes les 100 frames
            self.reid.cleanup_old_tracks(self.frame_count)

        return frame

    def run(self):
        try:
            self.running = True
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(0.001, 1 / fps) if fps > 0 else 0.033

            # Main processing loop
            while self.running and self.cap.isOpened():
                start_time = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    break

                processed = self.process_frame()

                # Send frame to UI
                try:
                    self.frame_queue.put_nowait((self.vid_idx, processed))
                except queue.Full:
                    pass

                # Control frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0.0, frame_interval - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            print(f"Processor {self.source} crashed: {e}")

        finally:
            # Make sure resources are always released
            print(f"Processor {self.source} exited")
