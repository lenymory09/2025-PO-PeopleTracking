# Documentation caméra réseau

## Etapes

 1. Installer windows (ubuntu non fonctionnel) sur un pc

 2. Connecter vous au bon réseau 

 3. Installer OBS studio avec le .exe sur leur site web (https://obsproject.com/)

 5. mettre à jour le Programme d'installation d'application pour pouvoir utiliser winget

 4. Installer NDI grâce au github distroav. (copier la ligne de commande pour l'installation et la lancer dans un cmd)

 5. Créer un auto-logon à la session grâce à l'app autologon disponible sur cette description de vidéo (https://www.youtube.com/watch?v=oIyXCEzKvE0) donnée l'utilisateur et le mdp c'est tout

 6. Lancer obs à chaque démarrage de session en apppuiant sur Win + R → tape shell:startup puis créer un raccourcie et indique ce chemin de déstination (C:\Program Files\obs-studio\bin\64bit\obs64.exe
)

7. Active le bureau à distance dans les paramètres windows 

8. Donnée des IP fixes aux différants pc pour évité les problèmes par la suite et pouvoir utiliser le bureau à distance 

9. Modifier le bouton d'alimentation dans le panneau de config (afficher par grandes icônes,option d'alimentaion, choisir l'action des boutons d'alimentation, arrêt)

10. Allez dans les paramètres d'obs et changer la résolution de sortie

11. Allez dans les outils obs et cliquez sur NDI (cochez sortie et remplissez le champs nom)

12. créer une nouvelle source de votre choix et vérifier qu'elle s'affiche bien et que tout fonctionne 

13. Il suffit plus qu'a affiché la source sur le pc récepteur avec la source NDI et choisir le bon nom 



## Problèmes 

 * En cas de problème avec obs vérifier si le réseau bloque l'envoie du flux et si nécessaire créer un réseau interne pour empêcher cela 