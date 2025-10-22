# # utils.py

# def dire_bonjour(nom: str) -> str:
#     """Retourne un message de salutation personnalisé."""
#     return f"Bonjour, {nom} !"

# def addition(a: int, b: int) -> int:
#     """Retourne la somme de deux nombres."""
#     return a + b

# def est_pair(nombre: int) -> bool:
#     """Vérifie si un nombre est pair."""
#     return nombre % 2 == 0

# if __name__ == "__main__":
#     # Cette partie ne s'exécute que si tu lances utils.py directement
#     print(dire_bonjour("Test"))
#     print("5 + 3 =", addition(5, 3))
#     print("10 est pair ?", est_pair(10))

import mysql.connector

def connect_db():
    # Connexion à la base de données
    conn = mysql.connector.connect(
        host="localhost",      # ou l'IP de ton serveur MySQL
        user="root",
        password="Admlocal1",
        database="IA_DB"
    )

# def insert_personne_passage():
#     # Créer un curseur pour exécuter les requêtes
#     cursor = conn.cursor()
#     # Exemple : insérer une donnée
#     cursor.execute("INSERT INTO personne (ID_personne) VALUES (24)")
#     # Sauvegarder les changements
#     conn.commit()

def insert_personne_passage(ID_personne, ID_lieux):
    # Créer un curseur pour exécuter les requêtes
    cursor = conn.cursor()
    
    # Insérer la personne si elle n’existe pas déjà (facultatif, mais évite les erreurs de clé étrangère)
    cursor.execute("INSERT IGNORE INTO personne (ID_personne) VALUES (%s)", (ID_personne,))
    
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