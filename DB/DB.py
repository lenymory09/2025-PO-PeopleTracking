import mysql.connector

def connect_db():
    # Connexion à la base de données
    conn = mysql.connector.connect(
        host="localhost",      # ou l'IP de ton serveur MySQL
        user="root",
        password="Admlocal1",
        database="IA_DB"
    )

# créer la base de données
def create_db():
    # Créer un curseur pour exécuter les requêtes
    conn = mysql.connector.connect(
        host="localhost",      # ou l'IP de ton serveur MySQL
        user="root",
        password="Admlocal1"
    )
    cursor = conn.cursor()
    
    # créer la base de données
    cursor.execute("""        
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
    conn.commit()
    cursor.close() 
    conn.close()

# insérer une personne dans la base de données 
def insert_personne_passage(ID_personne, ID_lieux):
    # Créer un curseur pour exécuter les requêtes
    cursor = conn.cursor()
    
    # Insérer la personne
    cursor.execute("INSERT INTO personne (ID_personne) VALUES (%s)", (ID_personne,))
    
    # Insérer le passage dans la table visiter avec la date actuelle
    cursor.execute("""
        INSERT INTO visiter (ID_personne, ID_lieux, date_visite)
        VALUES (%s, %s, NOW())
    """, (ID_personne, ID_lieux))
    
    # Sauvegarder les changements
    conn.commit()

# # Exemple : lire des données
# cursor.execute("SELECT * FROM users")
# rows = cursor.fetchall()

# for row in rows:
#     print(row)

def close_db():
    # Fermer le curseur et la connexion
    cursor.close() 
    conn.close()