import queue
import time

import cv2
from ultralytics import YOLO
import numpy as np

from typing import Dict, Optional, List, Tuple
from queue import Queue

from ultralytics.engine.results import Boxes

from reid import EnhancedReID
from .deep_sort.deep_sort.detection import Detection
from .deep_sort.tools.generate_detections import extract_image_patch
from utils import draw_person_box, chrono

from tracking.deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from tracking.deep_sort.tools import generate_detections as gdet
from tracking.deep_sort.deep_sort import nn_matching

import torch


def filter_boxes_by_dimensions(boxes: List[Boxes], width: int, min_box_ratio, max_box_ratio, min_box_height,
                               max_box_height) -> List[Boxes]:
    return list(
        filter(lambda box: is_correct_box(box, width, min_box_ratio, max_box_ratio, min_box_height, max_box_height),
               boxes))


def is_correct_box(box: Boxes, width: int, min_box_ratio, max_box_ratio, min_box_height, max_box_height) -> bool:
    """
    Vérifie si la boîte donné en paramètre est correcte et apte à être utilisé.

    Args:
        :param max_box_height: Taille max de la boîte
        :param min_box_height: Taille minimum de la boîte
        :param max_box_ratio: Ratio max de la boite
        :param min_box_ratio: Ratio min de la boite
        :param box: Boite à analyser
        :param width: largeur de la frame analysé
    Returns:
        :returns: vrai si la boite est juste et faux sinon.
    """
    x1, y1, x2, y2 = box.xyxy[0]
    width_box = x2 - x1
    height_box = y2 - y1
    # conf_correct = box.conf[0] > 0.60
    # is_dimensions_correct = width_box > 100 and height_box > 120
    ratio = height_box / width_box
    print(min_box_height, max_box_height)
    return bool(
        x1 > 25 and x2 < width - 25 and min_box_ratio < ratio < max_box_ratio and min_box_height < height_box < max_box_height)  # and is_dimensions_correct


class Track:
    """
    Classe représentant un Track DeepSORT
    """
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox

class DeepSortWrapper:
    """
    Classe qui gère l'algorithme DeepSORT dans le projet.
    """
    def __init__(self, model_filename='models/mars-small128.pb', max_cosine_distance=0.4, nn_budget=None):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        self.tracks: List[Track] = []

    @chrono
    def update(self, frame, detections):
        """
        Met à jour les tracks
        :param frame: image à analyser
        :param detections: détections des personnes
        """
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
            active_tracks.append(Track(track_id, bbox))

        self.tracks = active_tracks


class Camera:
    def __init__(self, source: str, reid: EnhancedReID, config, vid_idx):
        self.running = False
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.reid = reid
        self.frame_count = 0
        self.min_box_height = config['detection']['min_height_box'][vid_idx]
        self.max_box_height = config['detection']['max_height_box'][vid_idx]
        self.detection_config = config['detection']
        self.tracker_config = config['tracker']
        self.detection_device = torch.device(self.detection_config['device'])
        self.yolo = YOLO(config['models']['yolo'])
        self.yolo.to(self.detection_device)
        self.frame_queue: Optional[Queue] = None
        self.vid_idx = vid_idx
        self.ultracker = DeepSortWrapper(
            model_filename=config['tracker']['model'],
            max_cosine_distance=config['tracker']['max_cosine_distance'],
            nn_budget=config['tracker']['nn_budget']
        )
        self.ultrackid_to_pid: Dict[int, int] = {}
        self.current_persons = []

        self.passages_entrees = {}

    def get_tracked_pids(self) -> List[int]:
        """
        :return: Retourne les ids des tracks actifs
        """
        return list(filter(lambda track: track is not None,
                           map(lambda track: self.ultrackid_to_pid.get(track.track_id, None), self.ultracker.tracks)))

    @chrono
    def generate_detections(self, frame: np.ndarray) -> Tuple[List[Tuple], List[Boxes]]:
        """
        Generate the detections for the ReID Algorithm.
        :param frame: Frame to analyse
        :return: The detections and the boxes
        """
        _, frame_width, _ = frame.shape

        results = self.yolo.predict(
            frame,
            classes=[self.detection_config['person_class_id']],
            conf=self.detection_config['confidence_threshold'],
            verbose=False
        )[0]
        detections = []
        boxes = results.boxes
        boxes = filter_boxes_by_dimensions(
            boxes, frame_width, self.detection_config['min_box_ratio'],
            self.detection_config['max_box_ratio'], self.min_box_height,
            self.max_box_height
        )

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            # Format Deepsort RealTime: [left, top, right, bottom, confidence]
            # Deepsort Wrapper
            detections.append([x1, y1, x2, y2, conf])
        return detections, boxes

    def process_frame(self, frame) -> Optional[np.ndarray]:
        """
        Fait le traitement pour l'image courrante
        :param frame: image du flux
        :return: l'image venant du flux
        """
        height, width, _ = frame.shape
        self.frame_count += 1

        detections, boxes = self.generate_detections(frame)

        self.ultracker.update(frame, detections)
        tracks = self.ultracker.tracks

        # tracks = self.tracker.update_tracks(detections, frame=frame)
        bboxes = []
        crops = []

        # Process confirmed tracks
        for trk in tracks:
            l, t, r, b = trk.bbox
            bboxes.append(trk.bbox)
            bbox = (l, t, r - l, b - t)
            crops.append(extract_image_patch(frame, bbox, (256, 128)))

        if crops:
            features = self.reid.extract_features(crops)
        else:
            features = None

        assigned_ids = []
        if crops:
            for track, feat, bbox in zip(tracks, features, bboxes):
                tracked_pids = self.get_tracked_pids()
                feat = feat.detach().cpu().numpy()
                if track.track_id in self.ultrackid_to_pid and self.ultrackid_to_pid[
                    track.track_id] not in assigned_ids:
                    pid = self.ultrackid_to_pid[track.track_id]
                    assigned_ids.append(pid)

                    self.reid.update_gallery(pid, feat)
                else:
                    # Pour les nouveaux tracks, utiliser le premier feature
                    pid = self.reid.match_person(feat, assigned_ids)

                    if pid in tracked_pids:
                        pid = None
                    else:
                        self.ultrackid_to_pid[track.track_id] = pid

                if pid is None:
                    # Fallback pour debugging
                    print(f"Track {track.track_id} n'a pas de PID assigné")
                    draw_person_box(frame, track.bbox, "Unknown", (0, 0, 255))
                # Only draw the box if there is a valid pid that was found
                else:
                    x1, y1, x2, y2 = track.bbox
                    print(f"Camera {self.vid_idx} : {pid} : {x2 - x1} / {y2 - y1}")
                    label = self.reid.generate_label(pid)
                    color = self.reid.tracked_persons[pid]['color']
                    draw_person_box(frame, bbox, label, color)
            self.current_persons = assigned_ids

        return frame

    def release(self):
        self.cap.release()

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
            processed = self.process_frame(frame)

            # Send frame to UI
            try:
                self.frame_queue.put_nowait((self.vid_idx, processed))
            except queue.Full:
                pass

            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0.0, frame_interval - elapsed)
            time.sleep(sleep_time)

    def run_tracking(self):
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
                processed = self.process_tracking(frame)

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

    def update_passages(self, tracks, features):
        for track, feat in zip(tracks, features):
            passage = self.passages_entrees.get(track.track_id)
            feat = feat.detach().cpu().numpy()
            if passage is None:
                passage = {
                    "features": np.array([feat]),
                }
                self.passages_entrees[track.track_id] = passage
            else:
                track_feats = passage['features']
                passage['features'] = np.vstack((track_feats[-(100 - 1):], feat))


    def process_tracking(self, frame):
        height, width, _ = frame.shape
        self.frame_count += 1

        detections, boxes = self.generate_detections(frame)

        self.ultracker.update(frame, detections)
        tracks = self.ultracker.tracks

        # tracks = self.tracker.update_tracks(detections, frame=frame)
        bboxes = []
        crops = []

        # Process confirmed tracks
        for trk in tracks:
            l, t, r, b = trk.bbox
            bboxes.append(trk.bbox)
            bbox = (l, t, r - l, b - t)
            crops.append(extract_image_patch(frame, bbox, (256, 128)))

        if crops:
            features = self.reid.extract_features(crops)
        else:
            features = None


        if crops:
            self.update_passages(tracks, features)
            for track, feat, bbox in zip(tracks, features, bboxes):
                x1, y1, x2, y2 = track.bbox
                print(f"Camera {self.vid_idx} : {track.track_id} : {x2 - x1} / {y2 - y1}")
                label = f"Passage {track.track_id}"
                draw_person_box(frame, bbox, label, (0,0,255))

        return frame