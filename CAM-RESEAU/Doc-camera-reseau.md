# Documentation caméra réseau

## Etapes


 1. Installer windows (ubuntu non fonctionnel) sur un pc


 2. Connecter vous au bon réseau grâce au routeur que vous avez branché et allumé


 3. Installer OBS studio sur windows avec le .exe sur leur site web (https://obsproject.com/). Pour le mac il suffit de l'installer le apple.dmg


 4. Pour Installer DistroAV sur Mac il suffit de faire cette commande dans le terminale  
  ```bash
   brew install --cask distroav
```

 5. mettre à jour le Programme d'installation d'application pour pouvoir utiliser winget


 5. Installer NDI grâce au github distroav. Copier la ligne de commande pour l'installation et la lancer dans un cmd (lien du github https://github.com/DistroAV/DistroAV)


 6. Créer un auto-logon à la session grâce à l'app autologon disponible sur cette description de vidéo (https://www.youtube.com/watch?v=oIyXCEzKvE0) donnée l'utilisateur et le mdp c'est tout


 7. Lancer obs à chaque démarrage de session en apppuiant sur Win + R → tape shell:startup puis créer un raccourcie et indique ce chemin de déstination (C:\Program Files\obs-studio\bin\64bit\obs64.exe
)


8. Active le bureau à distance dans les paramètres windows 


9. Donnée des IP fixes aux différants pc pour évité les problèmes par la suite et pouvoir utiliser le bureau à distance 


10. Modifier le bouton d'alimentation dans le panneau de config (afficher par grandes icônes,option d'alimentaion, choisir l'action des boutons d'alimentation, arrêt)


11. Allez dans les paramètres d'obs et changer la résolution de sortie


12. Allez dans les outils obs et cliquez sur NDI (cochez sortie et remplissez le champs nom)


13. créer une nouvelle source de votre choix et vérifier qu'elle s'affiche bien et que tout fonctionne 


14. Il suffit plus qu'a affiché la source sur le pc récepteur avec la source NDI et choisir le bon nom 



## Problèmes 

 * En cas de problème avec obs vérifier si le réseau bloque l'envoie du flux et si nécessaire créer un réseau interne pour empêcher cela 

 * Si le flux ne s'affiche pas il faut se connecter en bureau à distance et vérifier que obs et bien lancé sur l'écran d'accueil 
