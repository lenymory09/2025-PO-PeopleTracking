# Person Tracker

---

## Introduction

---
Ce projet est une application de détection de tracking et de reconnaissance de personne multi-caméra avec IA permettant d'afficher 
le nombre de personnes qui vont devant la caméra et qui enregistre les heures de passage devant la caméra.

## Installation
Pour installer le projet voici les étapes :
1. Cloner / Télécharger le projet. :
```bash
git clone https://github.com/lenymory09/2025-PO-PeopleTracking
cd 2025-PO-PeopleTracking
```
2. Créer un environement conda (si conda n'est pas installé, installez le svp).
```bash
conda create --name people_tracking python=3.10
conda activate people_tracking
pip install -r requirements.txt
```
3. Téléchargement des modèles pour la ReID et la détection.
```bash
python download_models.py
```

4. Si vous êtes sur un PC Nvidia veuillez installer cuda pour la rapidité. Et installer les librairies pour :
```bash

```
5. 


## Algorithmes utilisés

- Détection : YOLO ([Ultralytics](https://github.com/ultralytics/ultralytics))
- Tracking : [DeepSORT](https://github.com/nwojke/deep_sort)
- Reconnaissance (ReID) : OSNet AIN ([TorchReID](https://github.com/KaiyangZhou/deep-person-reid))

## Fonctionnement
1. Lecture de l'image de la caméra.
2. Détection des personnes dans l'image avec YOLO.
3. Génération des features (vecteurs mathématiques) des gens sur l'image avec OSNet AIN.
4. Tracking des personnes sur l'image par rapport aux features.
5. Si les personnes viennent d'arriver sur le plan alors il les réidentifie avec OSNet AIN.
6. Mise à jour de la galerie de features.
7. Dessiner des boîtes pour les personnes avec leur ID sur l'image
8. Affichage de l'image dans la GUI PySide6 (Qt)


## Structure du projet
- `DB/`: fichiers de la DB
- `gui/`: Fichiers de la GUI 
- `reid/`: Fichier de la reid 
- `tracking/`: contient les fichiers pour gérés le tracking et le traitement du flux.
- `videos_exemples/`: vidéos d'exemples
- `config.yaml` : Fichier de configuration du projet.
- `main.py`: Point d'entrée du programme.
- `models/` : Répertoire qui contient les modèles pré-entrainés

## Remerciements
Ce projet est basé sur plusieurs autres projets :
- https://github.com/baseershah7/Multi_Camera_Surveillance_Person_Reid_Tracking
- https://github.com/nwojke/cosine_metric_learning
- https://github.com/nwojke/deepsort
- https://github.com/KaiyangZhou/deep-person-reid
- https://medium.com/@serurays/object-detection-and-tracking-using-yolov8-and-deepsort-47046fc914e9