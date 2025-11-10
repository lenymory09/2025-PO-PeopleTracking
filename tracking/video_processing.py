import queue
import time

from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from ultralytics import YOLO
import numpy as np

from typing import Dict, Optional, List, Tuple
from queue import Queue

from ultralytics.engine.results import Boxes

from reid import EnhancedPersonTracker
from .deep_sort.deep_sort.detection import Detection
from .deep_sort.tools.generate_detections import extract_image_patch
from utils import draw_person_box, chrono

from tracking.deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from tracking.deep_sort.tools import generate_detections as gdet
from tracking.deep_sort.deep_sort import nn_matching

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
    width_box = x2 - x1
    height_box = y2 - y1
    conf_correct = box.conf[0] > 0.70
    # is_dimensions_correct = width_box > 100 and height_box > 120
    ratio = height_box / width_box
    return bool(x1 > 25 and x2 < width - 25 and 2 < ratio < 3.5)  # and is_dimensions_correct


class Track:
    def __init__(self, track_id, bbox, features):
        self.track_id = track_id
        self.bbox = bbox
        self.features = features


class DeepSortWrapper:
    def __init__(self, model_filename='models/mars-small128.pb', max_cosine_distance=0.4, nn_budget=None):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        self.tracks: List[Track] = []

    @chrono
    def update(self, frame, detections):

        # Step 1: If no detections, run a predict-update cycle with an empty list.
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self._update_tracks()
            return

        # Step 2: Convert [x1, y1, x2, y2] to [x, y, w, h] for Deep SORT
        bboxes = np.array([d[:4] for d in detections])
        scores = [d[4] for d in detections]
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]

        # Step 3: Generate appearance features for each bounding box
        features = self.encoder(frame, bboxes)

        # Step 4: Wrap everything in Deep SORT's Detection objects
        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        # Step 5: Predict and update the tracker
        self.tracker.predict()
        self.tracker.update(dets)
        self._update_tracks()

    def _update_tracks(self):
        active_tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()  # returns [x1, y1, x2, y2]
            track_id = track.track_id
            features = track.features
            active_tracks.append(Track(track_id, bbox, features))

        self.tracks = active_tracks


class Camera:
    def __init__(self, source: str, reid: EnhancedPersonTracker, config, vid_idx):
        self.running = False
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.reid = reid
        self.frame_count = 0
        # self.tracker = DeepSort(
        #     max_age=config['tracker']['max_age'],
        #     n_init=config['tracker']['n_init'],
        #     max_cosine_distance=config['tracker']['max_cosine_distance'],
        #     nn_budget=config['tracker']['nn_budget'],
        # )
        self.detection_config = config['detection']
        self.yolo = YOLO(config['models']['yolo'])
        self.frame_queue: Optional[Queue] = None
        self.vid_idx = vid_idx
        self.ultracking = DeepSortWrapper(config['tracker']['model'])
        self.ultrackid_to_pid: Dict[int, int] = {}

    @chrono
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
        boxes = filter_boxes_by_dimensions(boxes, frame_width)
        # boxes = filter_boxes_by_overlag(boxes)
        # boxes = filter_by_aspect_ratio(boxes, 0.4)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            # Format: ([left, top, width, height], confidence, class_id)
            # detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))
            detections.append([x1, y1, x2, y2, conf])
        return detections, boxes

    def read(self):
        return self.cap.read()

    def generate_crops(self, frame: np.ndarray, bboxes: Tuple[int, int, int, int]) -> List[np.ndarray]:
        crops = []
        for (l, t, r, b) in bboxes:
            crops.append(frame[l:r, t:b])

        return crops

    def process_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.read()
        if not ret:
            return None

        height, width, _ = frame.shape
        self.frame_count += 1

        detections, boxes = self.generate_detections(frame)

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            box_width = x2 - x1
            box_height = y2 - y1
            print(box_height / box_width)

        self.ultracking.update(frame, detections)
        tracks = self.ultracking.tracks
        bboxes = []
        crops = []

        for trk in tracks:
            l, t, r, b = map(int, trk.bbox)
            # if (r - l) > 10 and (b - t) > 10:
            bboxes.append((l, t, r, b))
            bbox = (l, t, r - l, b - t)
            crops.append(extract_image_patch(frame, bbox, (100, 200)))

        assigned_ids = []

        # features = self.reid.generate_embeddings(frame, bboxes)
        if crops:
            # features = self.reid.extract_features(crops)
            # for track, feat in zip(tracks, features):
            for track in tracks:
                if track.track_id in self.ultrackid_to_pid:
                    pid = self.ultrackid_to_pid[track.track_id]
                    assigned_ids.append(pid)

                    if len(track.features) > 0:
                        # print("track trouvé")
                        feat = track.features[-1]  # Moyenne sur tous les features
                        # embed = embed.flatten()  # IMPORTANT: flatten() retourne une nouvelle array
                        self.reid.update_gallery(pid, feat)
                else:
                    # Pour les nouveaux tracks, utiliser le premier feature
                    feat = np.array(track.features[-1])  # Premier feature seulement
                    # feat = feat.flatten()  # S'assurer que c'est 1-D

                    pid = self.reid.match_person(feat, assigned_ids)

                    self.ultrackid_to_pid[track.track_id] = pid

                # Only draw the box if there is a valid pid that was found
                if pid is not None:
                    label = self.reid.generate_label(pid)
                    color = self.reid.tracked_persons[pid]['color']
                    draw_person_box(frame, track.bbox, label, color)
                else:
                    # Fallback pour debugging
                    print(f"Track {track.track_id} n'a pas de PID assigné")
                    draw_person_box(frame, track.bbox, "Unknown", (0, 0, 255))

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
        # embeds = self.reid.generate_embeddings(frame, detections)
        # embeds = list(map(lambda embed: embed / np.linalg.norm(embed), embeds))

        # assigned_ids = []
        #
        # for current_embedding, box in zip(embeds, boxes):
        #     pid = self.reid.match_person(current_embedding, assigned_ids)
        #     label = self.reid.generate_label(pid)
        #     color = self.reid.tracked_persons[pid]['color']
        #     draw_person_box(frame, box.xyxy[0], label, color)

        return frame

    def run(self):
        # try:
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

    # except Exception as e:
    #     print(f"Processor {self.source} crashed: {e}")

    # finally:
    #     # Make sure resources are always released
    #     print(f"Processor {self.source} exited")
