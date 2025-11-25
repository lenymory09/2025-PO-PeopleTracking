import datetime
import cv2
import numpy as np
import torch
import torchreid
from scipy.spatial.distance import cosine
from typing import List
import os
from utils import euclidean_distance, chrono
import threading
from time import time

torch.set_num_threads(os.cpu_count())

MODELS_PATH = {
    "osnet_x1_0": "models/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    "osnet_ain_x1_0": "models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
}

class EnhancedReID:
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
        """
        Ajoute le nouvelle Embedding de la personne.
        :param pid: ID de la personne
        :param feat: Vecteur de la personne
        :return: 
        """
        with self.lock:
            entry = self.tracked_persons[pid]
            features = entry['features']
            entry['features'] = np.vstack((features[-(self.max_gallery_size - 1):], feat))

    def _create_id(self, feat):
        """
        Crée une nouvelle ID
        :param feat: Premier vecteur de la personne
        :return: Le nouveau ID créé
        """
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
        """
        Trouve la personne la plus proche dans la liste de personnes ou en crée une nouvelle ID.
        :param embed: vecteur de la personne.
        :param assigned_ids: ids assignés
        :return: la personne la plus proche.
        """
        matched_id = None
        best_score = float('inf')
        # print(self.tracked_persons)
        with self.lock:
            for known_id, person in self.tracked_persons.items():
                #person_embed = person['features'].mean(0)
                if known_id in assigned_ids:
                    continue
                # scores = []
                for ref_emb in person['features'][-20:]:  # Utilise des vecteurs récents
                    cosine_dist = cosine(embed, ref_emb)
                    euclidean_dist = euclidean_distance(embed, ref_emb)

                    # Combined score
                    score = 0.7 * float(cosine_dist) + 0.3 * (euclidean_dist / 10.0)
                    #score = cosine_dist
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
            if len(person['features']) >= 50 and not person['saved']:
                person['saved'] = True
                person['confirmed'] = True
                persons.append((pid, "confirmed",
                                datetime.datetime.fromtimestamp(person['timestamp']).strftime('%Y-%m-%d %H:%M:%S')))
        return persons

    def extract_features(self, bgr_imgs) -> List[np.ndarray]:
        batch = []
        for img in bgr_imgs:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch.append(img_rgb)

        return self.feature_extractor(batch)

    @staticmethod
    def calc_nb_persons(db_nb_personnes):
        calc_persons = db_nb_personnes * 0.9
        return round(calc_persons)

    def generate_label(self, pid: int) -> str:
        """
        Generate the label for the person Box
        :param pid: Person ID
        :return: the label generated
        """
        nb_embeddings = len(self.tracked_persons[pid]['features'])
        label = f"ID {pid} : {'Max' if nb_embeddings == self.max_gallery_size else nb_embeddings}"
        return label
