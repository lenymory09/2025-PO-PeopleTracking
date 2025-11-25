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

2. Créer un environnement conda et installer les dépendances python : 
```bash
conda create --name people_tracking python=3.10
conda activate people_tracking
pip install -r requirements.txt
```
3. Téléchargement des modèles pour la ReID et la détection.
```bash
python download_models.py
```

4. Cloner le dépot DeepSort.
```bash
cd tracking
git clone https://github.com/nwojke/deep_sort
cd ..
```

5. Créer un fichier .env à la racine avec le mdp et la db à utiliser :
```text
MYSQL_USER=root
MYSQL_PASSWORD=<mot de passe de la section>
MYSQL_DB=IA_DB
MYSQL_HOST=localhost
```

6. Installer la police `gui/fonts/ScienceGothic-Variable.ttf`

7. Modifier le config.yaml pour mettre les cameras :
```yaml
video:
  max_videos: 2
  sources: [0,1] # caméras
  # sources: ["videos_exemples/MOT17-1.webm", "videos_exemples/MOT17-2.webm"] # ou des vidéos
```

8. Lancer le projet.
```bash
python main.py
```

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
```
├──DB/   -> fichiers de la DB
├──gui/  -> Fichiers de la GUI (Lance le projet)
│   ├── app_gui.py -> Fichier contenant la structure de la GUI Qt de base.
│   ├── gui.py  -> gère GUI et lancement du projet.
│   ├── gui.ui  -> Qt Designer App
│   └── fonts/  -> Polices 
├──reid/        -> Scripts Reid (Permettant de gérer la reconnaissance des peronnes)
    ├──reid.py  -> 
├──tracking/ -> contient les fichiers pour gérés le tracking et le traitement du flux.
├──videos_exemples/ -> vidéos d'exemples (Multi Object Tracking challenge)
├──config.yaml -> Fichier de configuration du projet.
├──main.py     -> Point d'entrée du programme.
├──models/     -> Répertoire qui contient les modèles pré-entrainés
└──.env        -> Variable d'environnement (contient infos DB)
```
## Remerciements
Ce projet est basé sur plusieurs autres projets :
- https://github.com/baseershah7/Multi_Camera_Surveillance_Person_Reid_Tracking
- https://github.com/nwojke/cosine_metric_learning
- https://github.com/nwojke/deepsort
- https://github.com/KaiyangZhou/deep-person-reid
- https://medium.com/@serurays/object-detection-and-tracking-using-yolov8-and-deepsort-47046fc914e9