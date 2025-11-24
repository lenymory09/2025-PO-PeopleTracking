from typing import Optional

import mysql.connector
import dotenv
import os

from mysql.connector.abstracts import MySQLCursorAbstract

dotenv.load_dotenv()
print(".env chargé avec succès.")


class DB:
    def __init__(self):
        self.conn: Optional[mysql.connector.MySQLConnection] = None
        self.cursor: Optional[MySQLCursorAbstract] = None
        self.create_db()
        self.connect_db()

    # créer la base de données
    def create_db(self):
        # Créer un curseur pour exécuter les requêtes
        conn_root = mysql.connector.connect(
            host=os.environ['MYSQL_HOST'],  # ou l'IP de ton serveur MySQL
            user=os.environ['MYSQL_USER'],
            password=os.environ['MYSQL_PASSWORD'],
        )
        cursor_root = conn_root.cursor()

        # créer la base de données
        commands = [
            "CREATE DATABASE IF NOT EXISTS IA_DB",
            "USE IA_DB",
            "DROP TABLE IF EXISTS visites",
            """
            CREATE TABLE visites
            (
                ID          INT AUTO_INCREMENT PRIMARY KEY,
                id_personne INT          not null unique,
                state       varchar(255) not null,
                timestamp   timestamp
            )
            """,
        ]
        for command in commands:
            cursor_root.execute(command)

        # cursor_root.executemany("""
        #                     -- Création de la base de données
        #                     CREATE DATABASE IF NOT EXISTS IA_DB;
        #                     USE IA_DB;
        #                     -- Création de la table personne
        #                     /*DROP TABLE IF EXISTS personne;
        #                     CREATE TABLE personne
        #                     (
        #                         ID_pers     INT AUTO_INCREMENT PRIMARY KEY,
        #                         ID_personne INT NOT NULL UNIQUE
        #                     );*/
        #
        #                     DROP TABLE IF EXISTS visites;
        #                     CREATE TABLE visites
        #                     (
        #                         ID INT AUTO_INCREMENT PRIMARY KEY,
        #                         id_personne INT not null unique,
        #                         state varchar not null,
        #                         timestamp datetime
        #                     );
        #
        #                     -- Création de la table lieux
        #                     /*DROP TABLE IF EXISTS lieux;
        #                     CREATE TABLE lieux
        #                     (
        #                         ID_lieux INT AUTO_INCREMENT PRIMARY KEY,
        #                         lieux    VARCHAR(50) NOT NULL
        #                     );*/
        #
        #                     -- Création de la table visiter (relation N-N entre personne et lieux)
        #                     /*DROP TABLE IF EXISTS visiter;
        #                     CREATE TABLE visiter
        #                     (
        #                         ID_personne INT      NOT NULL,
        #                         ID_lieux    INT      NOT NULL,
        #                         date_visite DATETIME NOT NULL DEFAULT NOW(),
        #
        #                         PRIMARY KEY (ID_personne, ID_lieux, date_visite),
        #
        #                         CONSTRAINT fk_visiter_personne FOREIGN KEY (ID_personne) REFERENCES personne (ID_pers)
        #                             ON DELETE CASCADE
        #                             ON UPDATE CASCADE,
        #                         CONSTRAINT fk_visiter_lieux FOREIGN KEY (ID_lieux) REFERENCES lieux (ID_lieux)
        #                             ON DELETE CASCADE
        #                             ON UPDATE CASCADE
        #                     );*/""")

        # Sauvegarder les changements
        conn_root.commit()
        cursor_root.close()
        conn_root.close()

    def connect_db(self):
        # Connexion à la base de données
        self.conn = mysql.connector.connect(
            host=os.environ['MYSQL_HOST'],  # ou l'IP de ton serveur MySQL
            user=os.environ['MYSQL_USER'],
            password=os.environ['MYSQL_PASSWORD'],
            database=os.environ['MYSQL_DB']
        )
        self.cursor = self.conn.cursor()

    # insérer une les lieux dans la base de données
    def fill_DB(self):
        # Insérer les lieux dans la DB
        self.cursor.execute("INSERT INTO personne (ID_lieux, lieux) VALUES (1, stand)", )
        self.cursor.execute("INSERT INTO personne (ID_lieux, lieux) VALUES (2, porte)", )

        # Sauvegarder les changements
        self.conn.commit()

    def insert_visites(self, personnes):
        self.cursor.executemany("INSERT INTO visites (id_personne, state, timestamp) VALUES (%s,%s,%s)", personnes)
        self.conn.commit()

    def fetch_nb_personnes(self):
        self.cursor.execute("SELECT COUNT(*) as nb_personnes FROM visites")
        return self.cursor.fetchone()


    def fetch_personnes(self, ids):
        placeholders = ', '.join(['%s'] * len(ids))
        query = f"SELECT v.id_personne, TIME(v.timestamp) FROM visites v WHERE v.id_personne IN ({placeholders})"
        self.cursor.execute(query, ids)
        return self.cursor.fetchall()

    def close_db(self):
        # Fermer le curseur et la connexion
        self.cursor.close()
        self.conn.close()

# # Exemple : lire des données
# cursor.execute("SELECT * FROM users")
# rows = cursor.fetchall()

# for row in rows:
#     print(row)
