# *Doc pour les 2ème années en informatique*

## Résumé
- Dans le cadre de notre projet de portes ouvertes, nous avons réalisé, avec mes collègues de 3ᵉ année en informatique, un système de 	traçage, détection et réidentification de personnes basé sur l’intelligence artificielle. Ce document a pour but d’expliquer brièvement le 	fonctionnement de notre projet ainsi que les outils techniques utilisés. Merci de votre attention et bonne lecture !


## Mise en place du stand

### Caméra
 
1. Allumer les NUC avec les caméras connectées en USB.

2. Sur le Mac, lancer **OBS**. La source NDI devrait déjà être présente avec la scène configurée. Vérifier que le flux est actif, avec éventuellement quelques secondes de latence.

3. Activer la **Caméra Virtuelle** dans OBS(sur le mac).

4. Une fois le flux distant activé, connecter la caméra USB sur le Mac.
 
 
---
 
### Démarrage de la base de données
 
1. Prend l'image docker sur dockerhub (biensur docker doit être préinstallée)
``` bash  
 docker pull mysql:oraclelinux9
```
2. Crée le conteneurs de la db
``` bash
docker run --name mysql-porte-ouvert -p 3306:3306 -e MYSQL_ROOT_PASSWORD=Admlocal1 -d mysql:oraclelinux9
```
3. verifiée que le conteneurs tourne bien
 
``` bash
docker ps
```
 
###  Lancement du script Python
 
1. Ouvrir un terminal.
2. Activer l’environnement virtuel Conda :
   ```bash
   conda activate deepsort
   ```
 
 
3. Se rendre à la racine du projet :
   ```bash
   cd chemin/du/projet
   ```
 
 
4. Vérifier le fichier **config.yaml** dans le dossier `/src` :
   ```bash
   nano src/config.yaml
   ```
 
5. Dans `config.yaml`, vérifier que la configuration des sources vidéo est correcte. Par exemple :
   ```yaml
   # Video Processing
   video:
     max_videos: 4
     sources: [0,1]
   ```
   
    - Les chiffres `[0,1]` correspondent aux caméras à utiliser.
 
6. Lancer le script Python pour exécuter l’IA :
 
   ```bash
   python src/main.py
   ``` 

## Fonctionnement
- Le fonctionnement de notre projet se déroule en plusieurs étapes :

1. Information et conformité légale. Un document est affiché à l’entrée pour informer les visiteurs de l’utilisation et du traitement des données collectées, conformément à la nLPD.

![alt text](charte-entréedrawio.png)

2. Capture du flux vidéo. Les différentes caméras disposées dans la salle envoient un flux vidéo constant grâce à OBS (logiciel de streaming) et au plugin DistroAV (implémentation de la technologie NDI qui permet de transmettre de la vidéo et du son via le réseau).

3. Analyse et traitement de l’image. Le Mac reçoit le flux vidéo et l’analyse à l’aide des algorithmes YOLO (détection d’objets) et DeepSORT (suivi et identification).

4. Stockage des données. Une base de données enregistre les différents ID détectés, permettant ensuite d’effectuer des calculs (comme le nombre de personnes présentes ou le temps de présence).

5. Affichage du flux filtré. Le Mac affiche le flux vidéo filtré à travers une application qui combine les calculs et le flux vidéo, offrant une visualisation en temps réel des résultats.

## Limites

- Lors de la conception de notre projet, nous avons dû surmonter plusieurs obstacles :

1. Respect de la législation (nLPD) 



2. Il nous était impossible d’utiliser la reconnaissance faciale ou toute donnée biométrique sans recueillir le consentement écrit de chaque visiteur.

3. Puissance de calcul limitée

4. Le matériel utilisé est suffisant, mais nous avons atteint la limite de performances raisonnables pour un budget restreint.

5. Manque de temps

6. Nous avons manqué de temps pour affiner les réglages et optimiser entièrement notre système.

## Cas d'utilisation 

- **Surveillance intelligente : suivre des individus dans des zones sensibles (aéroports, gares) pour repérer des comportements anormaux sans intervention humaine constante.**

- **Recherche de personnes disparues : localiser une personne identifiée dans un réseau de caméras.**

- **Analyse d’affluence dans les centres commerciaux, musées, stades, transport en commun. (Optimisation d'espace)**

- **Suivi des employés dans les usines pour assurer la sécurité (zones interdites, détection de chutes, EPI non portés). (Gestion des accès automotasiés)**  

- **Prévention des accidents : détecter l’intrusion d’une personne sur une voie dangereuse.**


- (Réalité augmentée / VR : suivre fidèlement un utilisateur dans un espace.)
- (Installation artistique où la présence et le déplacement de personnes génèrent du contenu.)

## Type d'IA  

- Notre type d'IA est basé sur la vision par ordinateur (une branche de l'intelligence artificielle qui permet  d'analyser, d'interpréter et de comprendre le monde visuel à partir d'images et de vidéos)

- Ce type d'IA peut-être utilisé pour beaucoup de domaine par exemple : Smartphones & Apps

- Déverrouillage par reconnaissance faciale (tester sur un téléphone)

- Jeux vidéo & Réalité augmentée. Suivi de mouvement

- Commerce & Retail. Caisse automatique (Migros)

- Santé & Médecine. Analyse d’images médicales (scanner, IRM, radiologie), détection de cancers (peau, poumons…)

- Sécurité & Surveillance.Reconnaissance faciale, détection d’intrusion
 
- Automobile & Transport. Voitures autonomes (détection de piétons, lignes, panneaux), aide à la conduite (ADAS, freinage d’urgence, maintien de voie)
 
  
  ## Schéma de l'infrastructure de notre projet portes ouvertes 
  ![alt text](<schéma-fonctionnel 1.jpg>)
   ![alt text](IA.jpeg)
   ![alt text](code.png) 