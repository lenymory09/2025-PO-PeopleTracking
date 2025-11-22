import datetime

import cv2
import numpy as np
import torch
import torchreid
from torchvision import transforms
from scipy.spatial.distance import cosine
from typing import List
import os
from PIL import Image
from utils import euclidean_distance, chrono
import threading
from time import time

torch.set_num_threads(os.cpu_count())

MODELS_PATH = {
    "osnet_x1_0": "models/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    "osnet_ain_x1_0": "models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
}


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

        use_gpu = config['reid']['device'] in ["cuda", "mps"]
        model_name = config['models']['reid_gpu'] if use_gpu else config['models']['reid_cpu']

        # self.reid_model = torchreid.models.build_model(
        #    name=model_name,
        #    num_classes=1501,
        #    loss="softmax",
        #    pretrained=True,
        #    use_gpu=torch.backends.mps.is_available()
        # )

        # if model_name == "osnet_ain_x1_0":
        #    torchreid.utils.load_pretrained_weights(self.reid_model, 'models/osnet_ain_market1501_60.pth')

        # if use_gpu:
        #    self.reid_model.to(self.device)

        # self.reid_model.eval()

        self.lock = threading.Lock()

        self.feature_extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            model_path=MODELS_PATH[model_name],
            verbose=False,
            device=config['reid']['device']
        )

    def update_gallery(self, pid, feat):
        with self.lock:
            entry = self.tracked_persons[pid]
            features = entry['features']
            entry['features'] = np.vstack((features[-(self.max_gallery_size - 1):], feat))
            # entry['temp_frames'] = min(entry['temp_frames'] + 1, self.temporal_frames)
            # if len(entry['features']) > 2:
            #     pass
            #     print(cosine(feat, entry['features'][-2]))

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
                'color': color,
                'confirmed': False,
                'saved': False,
                'timestamp': int(time())
            }
            print(f"Created ID {pid} with color {color}.")
            return pid

    @chrono
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
                # euclidean_dist = euclidean_distance(embed, person_embed)

                # Combined score
                # score = 0.7 * float(cosine_dist) + 0.3 * (euclidean_dist / 10.0)
                score = cosine_dist
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

    def get_confirmed_persons(self):
        persons = []
        for pid, person in self.tracked_persons.items():
            if len(person['features']) >= 100 and not person['saved']:
                person['saved'] = True
                person['confirmed'] = True
                persons.append((pid, "confirmed",
                                datetime.datetime.fromtimestamp(person['timestamp']).strftime('%Y-%m-%d %H:%M:%S')))
        return persons

    # def generate_embeddings(self, frame: np.ndarray, boxes):
    #     result = []
    #     for box in boxes:
    #         result.append(self.extract_embedding(frame, box))
    #
    #     return result
    #
    # def extract_embedding(self, img: np.ndarray, box: Tuple):
    #     """Improved embedding extraction with preprocessing"""
    #     x1, y1, x2, y2 = map(int, box)
    #     # x1, y1, width, height = map(int, box)
    #     # x2, y2 = x1 + width, y1 + height
    #
    #     # Expand box slightly for better context
    #     padding = 10
    #     x1 = max(0, x1 - padding)
    #     y1 = max(0, y1 - padding)
    #     x2 = min(img.shape[1], x2 + padding)
    #     y2 = min(img.shape[0], y2 + padding)
    #
    #     person_crop = img[y1:y2, x1:x2]
    #     # Skip if crop is too small
    #     if person_crop.size == 0 or person_crop.shape[0] < 50 or person_crop.shape[1] < 25:
    #         return None
    #
    #     # Enhanced preprocessing
    #     person_crop = Image.fromarray(person_crop[:, :, ::-1])
    #
    #     # Apply multiple augmentations and average embeddings
    #     embeddings = []
    #     # for augment in [False, True]:
    #     #     if augment:
    #     #         tensor = self.transform(transforms.functional.hflip(person_crop)).unsqueeze(0)
    #     #     else:
    #     #         tensor = self.transform(person_crop).unsqueeze(0)
    #     #
    #     #     with torch.no_grad():
    #     #         emb = self.reid_model(tensor)
    #     #         embeddings.append(emb.squeeze().numpy())
    #     tensor = self.transform(person_crop).unsqueeze(0)
    #     with torch.no_grad():
    #         emb = self.reid_model(tensor)
    #         embeddings.append(emb.squeeze().numpy())
    #
    #     # Average the embeddings
    #     avg_embedding = np.mean(embeddings, axis=0)
    #     return avg_embedding / np.linalg.norm(avg_embedding)  # L2 normalize

    def extract_features(self, bgr_imgs) -> List[np.ndarray]:
        batch = []
        for img in bgr_imgs:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch.append(img_rgb)

        return self.feature_extractor(batch)

    def extract_features2(self, bgr_imgs) -> np.ndarray:
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

    @staticmethod
    def calc_nb_persons(db_nb_personnes):
        calc_persons = db_nb_personnes * 0.9
        return round(calc_persons / 5) * 5

    def generate_label(self, pid: int) -> str:
        """
        Generate the label for the person Box
        :param pid: Person ID
        :return: the label generated
        """
        nb_embeddings = len(self.tracked_persons[pid]['features'])
        label = f"ID {pid} : {'Max' if nb_embeddings == self.max_gallery_size else nb_embeddings}"
        return label
