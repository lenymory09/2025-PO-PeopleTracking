import queue
import time

from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from ultralytics import YOLO
import numpy as np

from typing import Dict, Optional, List, Tuple
from queue import Queue

from ultralytics.engine.results import Boxes

from person_tracker import EnhancedPersonTracker
from utils import draw_person_box

from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection

class Track:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox


class DeepSortWrapper:
    def __init__(self, model_filename='models/mars-small128.pb', max_cosine_distance=0.4, nn_budget=None):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        self.tracks = []

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
            active_tracks.append(Track(track_id, bbox))

        self.tracks = active_tracks


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
            draw_person_box(frame, box.xyxy[0], label, color)

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
