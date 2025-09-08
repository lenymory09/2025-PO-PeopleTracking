# Documentation installation bases de données projet porte ouvert

# Base de données utiliser 
- [image docker mysql ](https://hub.docker.com/_/mysql)

# Installation 
``` Attention docker dois être préinstaller !```
- ``` docker pull mysql:oraclelinux9 ```

# lancer et configurer la base de données
- ```docker run --name mysql-porte-ouvert -p 3306:3306 -e MYSQL_ROOT_PASSWORD=Admlocal1 -d mysql:oraclelinux9```
- se connecter avec MySQL Workbench
- créer une nouvelle connection entrer les identifiants et connecter vous a la DB
- entrer la script de création de la db
- ```sql
        -- Création de la base de données
    CREATE DATABASE IA_DB;
    USE IA_DB; 
        -- Création de la table personne
    CREATE TABLE personne (
        ID_pers INT AUTO_INCREMENT PRIMARY KEY,
        ID_personne INT NOT NULL UNIQUE
    );

    -- Création de la table lieux
    CREATE TABLE lieux (
        ID_lieux INT AUTO_INCREMENT PRIMARY KEY,
        lieux VARCHAR(50) NOT NULL
    );

    -- Création de la table visiter (relation N-N entre personne et lieux)
    CREATE TABLE visiter (
        ID_personne INT NOT NULL,
        ID_lieux INT NOT NULL,
        date_visite DATETIME NOT NULL DEFAULT NOW(),
        
        PRIMARY KEY (ID_personne, ID_lieux, date_visite),
        
        CONSTRAINT fk_visiter_personne FOREIGN KEY (ID_personne) REFERENCES personne(ID_pers)
            ON DELETE CASCADE
            ON UPDATE CASCADE,
        CONSTRAINT fk_visiter_lieux FOREIGN KEY (ID_lieux) REFERENCES lieux(ID_lieux)
            ON DELETE CASCADE
            ON UPDATE CASCADE
    );

  ```
