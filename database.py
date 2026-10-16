"""
Crea (se non esiste) il database SQLite del ristorante. Gira sul BE

  python database.py            -> crea le tabelle e inserisce i dati iniziali SOLO se mancano
                                   (le prenotazioni già salvate NON vengono toccate)
  python database.py --reset    -> cancella tutto e riparte da zero
"""
import os
import sqlite3
import sys

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ristorante.db")

MENU = [
    ('Antipasto', 'Bruschetta al Pomodoro e Basilico', 5.0),
    ('Antipasto', 'Tagliere di Salumi e Formaggi', 15.0),
    ('Primo', 'Spaghetti alla Carbonara', 12.0),
    ('Primo', 'Risotto ai Funghi Porcini', 14.0),
    ('Secondo', 'Bistecca Fiorentina (all etto)', 5.5),
    ('Dolce', 'Tiramisù Classico', 6.0),
    ('Bevanda', 'Acqua Naturale o Frizzante', 2.5),
]


INFO = [
    ('nome', 'Il Bocconcino osteria romana'),
    ('indirizzo', 'Via Ostilia 23, 00184 Roma'),
    ('orari', 'Aperto dal Martedì alla Domenica. Pranzo: 12:00 - 15:00. Cena: 19:00 - 23:00. Lunedì giorno di chiusura.'),
    ('contatti', 'Sito web: https://ilbocconcino.com'),
]


def init_db(reset: bool = False):
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Vecchio database rimosso.")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute('''CREATE TABLE IF NOT EXISTS prenotazioni (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        codice TEXT,
        cliente_anonimo TEXT,
        data_prenotazione TEXT,
        ora TEXT,
        numero_persone INTEGER,
        dati_cifrati TEXT)''')      # dati personali CIFRATI dal cliente (il BE non puo' leggerli)
    cur.execute('''CREATE TABLE IF NOT EXISTS menu (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        categoria TEXT, piatto TEXT, prezzo REAL)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS ristorante_info (
        chiave TEXT PRIMARY KEY, valore TEXT)''')

    # migrazione: vecchi DB senza la colonna 'dati_cifrati'
    if "dati_cifrati" not in [r[1] for r in cur.execute("PRAGMA table_info(prenotazioni)")]:
        cur.execute("ALTER TABLE prenotazioni ADD COLUMN dati_cifrati TEXT")

    if cur.execute("SELECT COUNT(*) FROM menu").fetchone()[0] == 0:  #evita duplicati
        cur.executemany("INSERT INTO menu (categoria, piatto, prezzo) VALUES (?, ?, ?)", MENU)
    cur.executemany("INSERT OR IGNORE INTO ristorante_info (chiave, valore) VALUES (?, ?)", INFO)

    conn.commit()
    conn.close()
    print(f"Database pronto: {DB_PATH}")


if __name__ == "__main__":
    init_db(reset="--reset" in sys.argv)  #cancellare vecchio database
