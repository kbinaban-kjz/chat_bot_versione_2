"""
Logica di prenotazione del BACKEND (gira solo sulla macchina del BE).

Il cliente è anonimo (CHAT/VOICE tolgono i dati personali): la prenotazione si
identifica con un CODICE di 5 cifre, dato alla conferma, necessario per
modificarla o annullarla anche in una conversazione successiva.

Disponibilità: una richiesta viene rifiutata ("il posto non è disponibile") se
  * esiste già una prenotazione con stesso giorno, stessa ora e stesso numero di
    persone (anche fatta da un altro dispositivo), oppure
  * la capienza massima della fascia oraria verrebbe superata.

Dati personali (GDPR): CHAT e VOICE mandano al BE solo una versione CIFRATA dei dati
personali, con una chiave che il BE non ha. Il BE la salva in 'dati_cifrati' ma non la
passa mai all'LLM e a chi chiede di rivedere i dati sensibili risponde MSG_GDPR.

Sicurezza: l'LLM propone, i tool decidono. conferma_prenotazione funziona solo se
la proposta è stata fatta in un turno PRECEDENTE (il cliente ha avuto modo di dire sì).
"""
import datetime as dt
import os
import random
import re
import sqlite3
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ristorante.db")
CAPIENZA = 40                 # coperti massimi per fascia oraria
MAX_PERSONE = 12              # oltre: contattare il ristorante
GIORNI_CHIUSURA = {0}         # lunedì
# fasce prenotabili: (primo orario, ultimo orario prenotabile)
FASCE = [(dt.time(12, 0), dt.time(14, 0)), (dt.time(19, 0), dt.time(22, 0))]
ORARI_TESTO = "pranzo 12:00-15:00 e cena 19:00-23:00 (chiuso il lunedì)"

GIORNI = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
MESI = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
        "agosto", "settembre", "ottobre", "novembre", "dicembre"]

MSG_GDPR = ("Mi dispiace, non posso fornire dati sensibili (nome, email, numero di telefono) "
            "a causa del GDPR. Posso invece aiutarti con la prenotazione: dimmi il codice di "
            "5 cifre per consultarla, modificarla o annullarla.")

STORIA_MAX = 12               # messaggi di dialogo ricordati per sessione (l'LLM non ha memoria)


def nuovo_stato() -> dict:
    return {"proposta": None,      # prenotazione/modifica proposta, in attesa del "sì" del cliente
            "fatte": [],           # codici delle prenotazioni confermate in questa sessione
            "storia": [],          # ultimi messaggi user/assistant (testo anonimo)
            "turno": 0,            # contatore dei messaggi ricevuti 
            "dati_cifrati": None}  

# UTILITA' 
def fmt_data(d: dt.date) -> str:
    return f"{GIORNI[d.weekday()]} {d.day} {MESI[d.month - 1]} {d.year}"


def _ora_valida(o: dt.time) -> bool:
    return any(a <= o <= b for a, b in FASCE)


#  DB --
def _nuovo_codice(conn) -> str:
    while True:
        codice = str(random.randint(10000, 99999))
        if not conn.execute("SELECT 1 FROM prenotazioni WHERE codice=?", (codice,)).fetchone():
            return codice


def _conn():
    """Connessione in autocommit (le transazioni si aprono con BEGIN IMMEDIATE)."""
    conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
    conn.execute("""CREATE TABLE IF NOT EXISTS prenotazioni (
        id INTEGER PRIMARY KEY AUTOINCREMENT, codice TEXT, cliente_anonimo TEXT,
        data_prenotazione TEXT, ora TEXT, numero_persone INTEGER, dati_cifrati TEXT)""")
    # migrazione: vecchi DB senza le colonne 'codice' e 'dati_cifrati'
    colonne = [r[1] for r in conn.execute("PRAGMA table_info(prenotazioni)")]
    if "codice" not in colonne:
        conn.execute("ALTER TABLE prenotazioni ADD COLUMN codice TEXT")
    if "dati_cifrati" not in colonne:
        conn.execute("ALTER TABLE prenotazioni ADD COLUMN dati_cifrati TEXT")
    for (pid,) in conn.execute("SELECT id FROM prenotazioni WHERE codice IS NULL").fetchall():
        conn.execute("UPDATE prenotazioni SET codice=? WHERE id=?", (_nuovo_codice(conn), pid))
    return conn


def _posti_occupati(conn, d: dt.date, o: dt.time, escludi: Optional[str] = None) -> int:
    """Coperti già prenotati nella stessa fascia (±1 ora e mezza)."""
    righe = conn.execute(
        "SELECT ora, numero_persone, codice FROM prenotazioni WHERE data_prenotazione=?",
        (d.isoformat(),)).fetchall()
    centro = o.hour * 60 + o.minute
    tot = 0
    for ora, n, codice in righe:
        if codice == escludi:
            continue
        h, m = map(int, ora.split(":"))
        if abs(h * 60 + m - centro) < 90:
            tot += n
    return tot


def _gia_prenotato(conn, d: dt.date, o: dt.time, persone: int,
                   escludi: Optional[str] = None) -> bool:
    """True se esiste già una prenotazione con stesso giorno, stessa ora e stesso
    numero di persone (fatta da qualunque dispositivo). 'escludi' = codice della
    prenotazione che si sta modificando, da non contare."""
    q = ("SELECT 1 FROM prenotazioni "
         "WHERE data_prenotazione=? AND ora=? AND numero_persone=?")
    par = [d.isoformat(), o.strftime("%H:%M"), persone]
    if escludi:
        q += " AND codice<>?"
        par.append(escludi)
    return conn.execute(q + " LIMIT 1", par).fetchone() is not None


def _slot_occupato(stato: dict, escludi: Optional[str] = None) -> bool:
    """Controllo anticipato (al riepilogo), prima di chiedere la conferma."""
    conn = _conn()
    try:
        return _gia_prenotato(conn, stato["data"], stato["ora"], stato["persone"], escludi)
    finally:
        conn.close()


def _prenota(stato: dict) -> Optional[str]:
    """Scrive la prenotazione in modo atomico. Ritorna il codice, None se non c'è posto."""
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")          # blocca altre scritture: niente overbooking
        if _gia_prenotato(conn, stato["data"], stato["ora"], stato["persone"]) \
                or _posti_occupati(conn, stato["data"], stato["ora"]) + stato["persone"] > CAPIENZA:
            conn.execute("ROLLBACK")
            return None
        codice = _nuovo_codice(conn)
        conn.execute(
            "INSERT INTO prenotazioni (codice, cliente_anonimo, data_prenotazione, ora, numero_persone,"
            " dati_cifrati) VALUES (?,?,?,?,?,?)",
            (codice, "anonimo", stato["data"].isoformat(),
             stato["ora"].strftime("%H:%M"), stato["persone"], stato.get("dati_cifrati")))
        conn.execute("COMMIT")
        return codice
    finally:
        conn.close()


def _aggiorna(codice: str, stato: dict) -> bool:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE") #gestire l'overbooking tra 2 utenti contemporanei , si blocca
        if _gia_prenotato(conn, stato["data"], stato["ora"], stato["persone"], escludi=codice) \
                or _posti_occupati(conn, stato["data"], stato["ora"], escludi=codice) + stato["persone"] > CAPIENZA:
            conn.execute("ROLLBACK")
            return False
        conn.execute(
            "UPDATE prenotazioni SET data_prenotazione=?, ora=?, numero_persone=? WHERE codice=?",
            (stato["data"].isoformat(), stato["ora"].strftime("%H:%M"), stato["persone"], codice))
        if stato.get("dati_cifrati"):             # il cliente ha dato nuovi dati personali (cifrati)
            conn.execute("UPDATE prenotazioni SET dati_cifrati=? WHERE codice=?",
                         (stato["dati_cifrati"], codice))
        conn.execute("COMMIT")
        return True
    finally:
        conn.close()


def _leggi(codici: list):
    if not codici:
        return []
    conn = _conn()
    try:
        return conn.execute(
            f"SELECT codice, data_prenotazione, ora, numero_persone FROM prenotazioni "
            f"WHERE codice IN ({','.join('?' * len(codici))}) ORDER BY data_prenotazione, ora",
            codici).fetchall()
    finally:
        conn.close()


def _elimina(codice: str) -> bool:
    conn = _conn()
    try:
        return conn.execute("DELETE FROM prenotazioni WHERE codice=?", (codice,)).rowcount > 0
    finally:
        conn.close()


def descrivi(riga) -> str:
    codice, data_iso, ora, n = riga
    return (f"Codice {codice}: {n} {'persona' if n == 1 else 'persone'}, "
            f"{fmt_data(dt.date.fromisoformat(data_iso))} alle {ora}")


def prenotazioni_sessione(stato: dict) -> list:
    """Per il contesto dell'LLM: prenotazioni di questa sessione, come testo."""
    return [descrivi(r) for r in _leggi(stato.get("fatte", []))]




# =========================================================== TOOL PER L'LLM ==
# Ogni tool ritorna un dict JSON-friendly. Convenzioni:
#   "ok": True/False
#   "messaggio": frase pronta in italiano (usata anche come riserva se l'LLM non
#                scrive niente dopo il tool)
# Se ok=False il messaggio spiega cosa manca/non va: l'LLM lo riformula al cliente.

_VUOTI = {"", "null", "none", "nil", "n/a", "na", "undefined", "unknown", "nessuno", "nessuna"}


def _arg(v):
    """Gli LLM piccoli scrivono "null"/"None" come stringa quando un parametro manca: per noi è vuoto."""
    return None if v is None or str(v).strip().lower() in _VUOTI else v


def _codice(v):
    """Codice prenotazione: '' se assente, None se presente ma non valido, altrimenti le 5 cifre."""
    v = _arg(v)
    if v is None:
        return ""
    cifre = re.sub(r"[\s-]", "", str(v))          # anche "4 8 2 1 3" dettato a voce
    return cifre if re.fullmatch(r"\d{5}", cifre) else None


_ERR_CODICE = "Il codice prenotazione deve essere di 5 cifre."


def _parse_data_iso(v) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(str(v).strip()[:10])
    except (ValueError, TypeError):
        return None


def _parse_ora_hhmm(v) -> Optional[dt.time]:
    """Accetta '20:00', '20.30', '20' (formati che l'LLM puo' produrre)."""
    try:
        s = str(v).strip().replace(".", ":")
        h, _, m = s.partition(":")
        return dt.time(int(h), int(m or 0))
    except (ValueError, TypeError):
        return None


def _parse_persone(v) -> Optional[int]:
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _errore(msg: str) -> dict:
    return {"ok": False, "messaggio": msg}


def _valida(persone, d, o, adesso: dt.datetime) -> Optional[str]:
    """Stesse regole di prima, ma ritorna il motivo del rifiuto (None = tutto ok)."""
    if persone is None:
        return "Manca il numero di persone."
    if not (1 <= persone <= MAX_PERSONE):
        return (f"Per {persone} persone il cliente deve contattare direttamente il ristorante: "
                f"online si accettano da 1 a {MAX_PERSONE} persone.")
    if d is None:
        return "Manca il giorno (o non è valido): usa il formato AAAA-MM-GG."
    if d < adesso.date():
        return "Quella data è già passata."
    if d.weekday() in GIORNI_CHIUSURA:
        return f"Il {GIORNI[d.weekday()]} il ristorante è chiuso. Orari: {ORARI_TESTO}."
    if o is None:
        return "Manca l'orario (o non è valido): usa il formato HH:MM."
    if not _ora_valida(o):
        return f"Alle {o.strftime('%H:%M')} la cucina è chiusa. Si può prenotare a {ORARI_TESTO}."
    if d == adesso.date() and dt.datetime.combine(d, o) < adesso:
        return "Quell'orario è già passato."
    return None


def _chiedi_quale(stato: dict, azione: str) -> Optional[dict]:
    """Senza codice: mostra le prenotazioni di questa conversazione e chiede QUALE
    (anche se ce n'è una sola, per non annullare/cambiare per un'frase vaga).
    None se in questa conversazione non ce ne sono."""
    righe = _leggi(stato["fatte"])
    if not righe:
        return None
    elenco = " ".join(descrivi(r) + "." for r in righe)
    if len(righe) == 1:
        msg = (f"In questa conversazione hai questa prenotazione: {elenco} "
               f"Vuoi {azione} questa prenotazione? Rispondi di sì o dimmi il codice di 5 cifre.")
    else:
        msg = f"Hai più prenotazioni: {elenco} Quale vuoi {azione}? Dimmi il codice di 5 cifre."
    return {"ok": True, "serve_scelta": True, "messaggio": msg}


def _riepilogo(persone: int, d: dt.date, o: dt.time) -> str:
    return f"tavolo per {persone} persone, {fmt_data(d)} alle {o.strftime('%H:%M')}"


def tool_controlla_disponibilita(stato: dict, persone=None, data=None, ora=None,
                                 adesso: Optional[dt.datetime] = None) -> dict:
    """Solo controllo: NON prenota e non cambia lo stato."""
    adesso = adesso or dt.datetime.now()
    p, d, o = _parse_persone(persone), _parse_data_iso(data), _parse_ora_hhmm(ora)
    motivo = _valida(p, d, o, adesso)
    if motivo:
        return _errore(motivo)
    conn = _conn()
    try:
        if _gia_prenotato(conn, d, o, p) or _posti_occupati(conn, d, o) + p > CAPIENZA:
            return {"ok": True, "disponibile": False,
                    "messaggio": f"Il posto non è disponibile per {_riepilogo(p, d, o)}."}
    finally:
        conn.close()
    return {"ok": True, "disponibile": True,
            "messaggio": f"Il posto è disponibile per {_riepilogo(p, d, o)}."}


def tool_proponi_prenotazione(stato: dict, persone=None, data=None, ora=None,
                              adesso: Optional[dt.datetime] = None) -> dict:
    """Valida e controlla la disponibilità; se ok salva la PROPOSTA nello stato.
    Non scrive nel DB: serve poi il 'sì' del cliente e conferma_prenotazione."""
    adesso = adesso or dt.datetime.now()
    p, d, o = _parse_persone(persone), _parse_data_iso(data), _parse_ora_hhmm(ora)
    motivo = _valida(p, d, o, adesso)
    if motivo:
        return _errore(motivo)
    stato2 = {"data": d, "ora": o, "persone": p}
    if _slot_occupato(stato2):
        return _errore(f"Il posto non è disponibile: per {p} persone il {fmt_data(d)} "
                       f"alle {o.strftime('%H:%M')} risulta già prenotato. "
                       "Proponi un altro orario o un altro giorno.")
    stato["proposta"] = {"modifica": None, "persone": p, "data": d, "ora": o,
                         "turno": stato["turno"]}
    return {"ok": True,
            "messaggio": f"Riepilogo: {_riepilogo(p, d, o)}. Confermi la prenotazione?",
            "prossimo_passo": "Riepiloga al cliente e chiedi conferma. NON confermare ora."}


def tool_modifica_prenotazione(stato: dict, codice=None, persone=None, data=None, ora=None,
                               adesso: Optional[dt.datetime] = None) -> dict:
    """Propone la modifica di una prenotazione esistente (serve il codice)."""
    adesso = adesso or dt.datetime.now()
    persone, data, ora = _arg(persone), _arg(data), _arg(ora)
    codice = _codice(codice)
    if codice is None:
        return _errore(_ERR_CODICE)
    if not codice and len(stato["fatte"]) == 1:
        codice = stato["fatte"][0]
    if not codice:
        scelta = _chiedi_quale(stato, "modificare")
        if scelta:
            return scelta
        return _errore("Per modificare serve il codice di 5 cifre dato alla conferma.")
    righe = _leggi([codice])
    if not righe:
        return _errore(f"Non trovo nessuna prenotazione con codice {codice}.")
    _, data_iso, ora_old, n_old = righe[0]
    if persone is None and data is None and ora is None:
        return {"ok": True, "messaggio": f"{descrivi(righe[0])}. Cosa vuoi cambiare: data, orario o numero di persone?"}
    p = _parse_persone(persone) if persone is not None else n_old
    d = _parse_data_iso(data) if data is not None else dt.date.fromisoformat(data_iso)
    o = _parse_ora_hhmm(ora) if ora is not None else dt.time(*map(int, ora_old.split(":")))
    motivo = _valida(p, d, o, adesso)
    if motivo:
        return _errore(motivo)
    if _slot_occupato({"data": d, "ora": o, "persone": p}, escludi=codice):
        return _errore(f"Il posto non è disponibile: per {p} persone il {fmt_data(d)} "
                       f"alle {o.strftime('%H:%M')} risulta già prenotato. "
                       "Proponi un altro orario o un altro giorno.")
    stato["proposta"] = {"modifica": codice, "persone": p, "data": d, "ora": o,
                         "turno": stato["turno"]}
    return {"ok": True,
            "messaggio": f"Riepilogo della modifica alla prenotazione {codice}: {_riepilogo(p, d, o)}. Confermi?",
            "prossimo_passo": "Riepiloga al cliente e chiedi conferma. NON confermare ora."}


def tool_conferma_prenotazione(stato: dict) -> dict:
    """Scrive nel DB la proposta in sospeso. Solo se il cliente ha avuto un turno per dire sì."""
    prop = stato.get("proposta")
    if not prop:
        return _errore("Non c'è nessuna prenotazione in attesa di conferma: "
                       "raccogli prima persone, giorno e ora e usa proponi_prenotazione.")
    if prop["turno"] == stato["turno"]:
        err = _errore("Il cliente non ha ancora confermato: chiedigli conferma e "
                      "attendi la sua risposta prima di richiamare questo tool.")
        err["bloccata"] = True
        return err
    dati = {"persone": prop["persone"], "data": prop["data"], "ora": prop["ora"],
            "dati_cifrati": stato.get("dati_cifrati")}
    stato["proposta"] = None
    if prop["modifica"]:
        cod = prop["modifica"]
        if not _aggiorna(cod, dati):
            return _errore("Il posto non è disponibile per quel giorno e orario. "
                           "Chiedi se vuole provare con un altro orario.")
        return {"ok": True, "codice": cod,
                "messaggio": f"Prenotazione modificata! Codice {cod}: {_riepilogo(dati['persone'], dati['data'], dati['ora'])}."}
    cod = _prenota(dati)
    if cod is None:
        return _errore("Il posto non è disponibile per quel giorno e orario. "
                       "Chiedi se vuole provare con un altro orario.")
    stato["fatte"].append(cod)
    return {"ok": True, "codice": cod,
            "messaggio": (f"Prenotazione confermata! Il tuo codice è {cod}: conservalo per modificare o "
                          f"annullare. {_riepilogo(dati['persone'], dati['data'], dati['ora']).capitalize()}. "
                          "Ti aspettiamo!")}


def tool_annulla_prenotazione(stato: dict, codice=None) -> dict:
    """Elimina una prenotazione già confermata. Senza codice chiede QUALE annullare."""
    codice = _codice(codice)
    if codice is None:
        return _errore(_ERR_CODICE)
    if not codice:
        scelta = _chiedi_quale(stato, "annullare")
        if scelta:
            return scelta
        return _errore("Per annullare serve il codice di 5 cifre dato alla conferma.")
    if not _elimina(codice):
        return _errore(f"Non trovo nessuna prenotazione con codice {codice}. "
                       "Controlla il codice che ti ho dato alla conferma.")
    if codice in stato["fatte"]:
        stato["fatte"].remove(codice)
    if stato.get("proposta") and stato["proposta"].get("modifica") == codice:
        stato["proposta"] = None
    return {"ok": True, "messaggio": f"Ho annullato la prenotazione con codice {codice}."}


def tool_annulla_richiesta_in_corso(stato: dict) -> dict:
    """Il cliente ci ripensa PRIMA di confermare: si butta la proposta, il DB non cambia."""
    if not stato.get("proposta"):
        return _errore("Non c'è nessuna richiesta in attesa di conferma: non hai annullato niente. "
                       "Se il cliente vuole cancellare una prenotazione già confermata devi usare "
                       "annulla_prenotazione.")
    stato["proposta"] = None
    return {"ok": True, "messaggio": "Va bene, ho annullato la richiesta. Posso aiutarti in altro modo?"}


def tool_consulta_prenotazioni(stato: dict, codice=None) -> dict:
    """Mostra una prenotazione (per codice) o quelle fatte in questa conversazione."""
    codice = _codice(codice)
    if codice is None:
        return _errore(_ERR_CODICE)
    if codice:
        righe = _leggi([codice])
        return ({"ok": True, "messaggio": descrivi(righe[0]) + "."} if righe
                else _errore(f"Non trovo nessuna prenotazione con codice {codice}."))
    righe = _leggi(stato["fatte"])
    if righe:
        msg = "Ecco le prenotazioni fatte in questa conversazione: " + " ".join(descrivi(r) + "." for r in righe)
    else:
        msg = ("In questa conversazione non hai ancora confermato nessuna prenotazione. "
               "Per privacy non conservo i nomi dei clienti: se hai un codice di prenotazione, dimmelo.")
    prop = stato.get("proposta")
    if prop:
        msg += (f" C'è anche una richiesta non ancora confermata: "
                f"{_riepilogo(prop['persone'], prop['data'], prop['ora'])}.")
    return {"ok": True, "messaggio": msg}


def tool_dati_personali_non_disponibili(stato: dict) -> dict:
    """GDPR: l'LLM lo chiama se il cliente vuole vedere nome/email/telefono (suoi o altrui)."""
    return {"ok": False, "gdpr": True, "messaggio": MSG_GDPR}
