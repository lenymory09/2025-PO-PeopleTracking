import mysql.connector


class DB:
    def __init__(self):
        self.create_db()
        self.connect_db()

    # créer la base de données
    def create_db(self):
        # Créer un curseur pour exécuter les requêtes
        conn_root = mysql.connector.connect(
            host="localhost",      # ou l'IP de ton serveur MySQL
            user="root",
            password="Admlocal1"
        )
        cursor_root = conn_root.cursor()
        
        # créer la base de données
        cursor_root.execute("""        
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
                        );""")
        # Sauvegarder les changements
        conn_root.commit()
        cursor_root.close() 
        conn_root.close()


    def connect_db(self):
        # Connexion à la base de données
        self.conn = mysql.connector.connect(
            host="localhost",      # ou l'IP de ton serveur MySQL
            user="root",
            password="Admlocal1",
            database="IA_DB"
        )
        self.cursor = self.conn.cursor()

    # insérer une les lieux dans la base de données 
    def fill_DB(self):
        # Insérer les lieux dans la DB
        self.cursor.execute("INSERT INTO personne (ID_lieux, lieux) VALUES (1, stand)",)
        self.cursor.execute("INSERT INTO personne (ID_lieux, lieux) VALUES (2, porte)",)
        
        # Sauvegarder les changements
        self.conn.commit()


    # insérer une personne dans la base de données 
    def insert_personne_passage(self, ID_personne, ID_lieux):
        # Insérer la personne
        self.cursor.execute("INSERT INTO personne (ID_personne) VALUES (%s)", (ID_personne,))
        
        # Insérer le passage dans la table visiter avec la date actuelle
        self.cursor.execute("""
            INSERT INTO visiter (ID_personne, ID_lieux, date_visite)
            VALUES (%s, %s, NOW())
        """, (ID_personne, ID_lieux))
        
        # Sauvegarder les changements
        self.conn.commit()

# # Exemple : lire des données
# cursor.execute("SELECT * FROM users")
# rows = cursor.fetchall()

# for row in rows:
#     print(row)


def close_db(self):
    # Fermer le curseur et la connexion 
    self.cursor.close() 
    self.conn.close()