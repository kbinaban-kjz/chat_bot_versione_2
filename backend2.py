"""
BACKEND (versione con TOOL)
Privacy: il BE non riceve mai dati personali in chiaro. Se il client ne ha, manda un
blob CIFRATO ("dati_cifrati") che il BE salva nel DB senza poterlo decifrare e che
NON viene mai passato all'LLM.

REQUISITO: usare ollama pull llama3.1
"""
import datetime
import json
import os
import re
import sqlite3
import threading
import time
from typing import Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel

import prenotazioni as pren

# CONFIG
DB_PATH = pren.DB_PATH                             
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1")    # modello locale CON supporto tool
# modello per le domande di cultura generale (può essere uno più grande/preciso, es. qwen2.5:14b)
MODEL_GENERALE = os.getenv("MODEL_GENERALE", MODEL_NAME)
LLM_TIMEOUT = 120
SESSIONE_TTL = 2 * 3600                             # le sessioni inattive vengono dimenticate dopo 2 ore
MAX_GIRI_TOOL = 5                                   # max chiamate consecutive ai tool per ogni messaggio

app = FastAPI(title="Backend ristorante")


class UserRequest(BaseModel):
    message: str
    session_id: str = "default"      # CHAT e VOICE mandano un id diverso per ogni conversazione
    dati_cifrati: Optional[str] = None   # dati personali CIFRATI dal client (il BE non ha la chiave)


# SESSIONI
_sessioni: dict = {}
_lock = threading.Lock()  # impedisce che due richieste simultanee modifichino il dizionario


def get_stato(session_id: str) -> dict:
    """Stato di una conversazione: prenotazione proposta, codici, storia del dialogo."""
    adesso = time.time()
    with _lock:
        for sid in [s for s, v in _sessioni.items() if adesso - v["ts"] > SESSIONE_TTL]:
            del _sessioni[sid]
        voce = _sessioni.setdefault(session_id, {"stato": pren.nuovo_stato(), "ts": adesso})
        voce["ts"] = adesso
        return voce["stato"]


# DATABASE
def leggi_db() -> dict:
    """Menu e informazioni del ristorante, letti dal DB a ogni richiesta (sempre aggiornati)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        menu = conn.execute("SELECT categoria, piatto, prezzo FROM menu ORDER BY id").fetchall()
        info = dict(conn.execute("SELECT chiave, valore FROM ristorante_info").fetchall())
        conn.close()
        return {"menu": menu, "info": info}
    except Exception as e:
        print(f"[DB] errore lettura: {e}")
        return {"menu": [], "info": {}}


def _prezzo(p: float) -> str:
    return f"{p:g}".replace(".", ",") + " euro"


def testo_menu(menu) -> str:
    if not menu:
        return "Il menu non è al momento disponibile."
    per_categoria = {}
    for cat, piatto, prezzo in menu:
        per_categoria.setdefault(cat, []).append(f"{piatto} ({_prezzo(prezzo)})")
    return "\n".join(f"{cat}: " + ", ".join(piatti) + "." for cat, piatti in per_categoria.items())


def testo_info(info: dict) -> str:
    return "\n".join(f"{k.capitalize()}: {v}" for k, v in info.items() if k != "nome")


# TOOL: menu e informazioni (quelli di prenotazione sono in prenotazioni.py)
def tool_get_menu(stato: dict, categoria=None) -> dict:
    menu = leggi_db()["menu"]
    if categoria:
        filtrato = [r for r in menu if str(categoria).lower() in r[0].lower()]
        menu = filtrato or menu            # categoria inesistente: meglio dare tutto il menu
    return {"ok": bool(menu), "messaggio": testo_menu(menu)}


def tool_get_info_ristorante(stato: dict) -> dict:
    info = leggi_db()["info"]
    return {"ok": bool(info), "nome": info.get("nome", ""),
            "messaggio": testo_info(info) or "Al momento non ho queste informazioni."}


PROMPT_GENERALE = """Sei un assistente cordiale e competente che risponde a domande di cultura generale. Rispondi SEMPRE in italiano, in massimo 2 o 3 frasi, senza elenchi, senza markdown e senza emoji (la risposta viene letta ad alta voce).
Dai SEMPRE una risposta utile con la tua migliore conoscenza: non rifiutarti e non rispondere "non ho informazioni" o "non lo so" a meno che tu non abbia davvero zero idea.
Se la domanda riguarda numeri o dati che cambiano nel tempo (quanti negozi, abitanti, prezzi, classifiche), dai la stima che ricordi usando "circa" o "intorno a" e aggiungi in poche parole che il dato potrebbe essere cambiato o non essere aggiornato.
Non inventare dettagli precisi che non ricordi (nomi, date esatte): in quel caso resta generico.
Rispondi direttamente alla domanda, senza parlare di ristoranti o di altro non richiesto."""


def tool_rispondi_domanda_generale(stato: dict) -> dict:
    """Domande che NON riguardano il ristorante: seconda chiamata all'LLM con un prompt corto e
    pulito (senza il contesto del ristorante, che confondeva il modello). Risponde alla frase
    originale del cliente, non a una riformulazione dell'LLM."""
    domanda = stato.get("ultimo_messaggio", "")
    messages = ([{"role": "system", "content": PROMPT_GENERALE}]
                + stato["storia"][-4:] + [{"role": "user", "content": domanda}])
    msg = _chiama_ollama(messages, usa_tool=False, modello=MODEL_GENERALE, temperatura=0.0)
    testo = _pulisci((msg or {}).get("content", ""))
    if not testo:
        return {"ok": False, "messaggio": "In questo momento non riesco a rispondere a questa domanda."}
    return {"ok": True, "risposta_diretta": True, "messaggio": testo}


def tool_mostra_mappa_ristorante(stato: dict) -> dict:
    """Chiede al client di mostrare la mappa DEL RISTORANTE (il client ha le coordinate)."""
    info = leggi_db()["info"]
    return {"ok": True, "mappa": True,
            "messaggio": f"Indirizzo: {info.get('indirizzo', 'non disponibile')}. Mappa mostrata al cliente."}


TOOL_FUNZIONI = {
    "get_menu": tool_get_menu,
    "get_info_ristorante": tool_get_info_ristorante,
    "mostra_mappa_ristorante": tool_mostra_mappa_ristorante,
    "rispondi_domanda_generale": tool_rispondi_domanda_generale,
    "controlla_disponibilita": pren.tool_controlla_disponibilita,
    "proponi_prenotazione": pren.tool_proponi_prenotazione,
    "modifica_prenotazione": pren.tool_modifica_prenotazione,
    "conferma_prenotazione": pren.tool_conferma_prenotazione,
    "annulla_prenotazione": pren.tool_annulla_prenotazione,
    "annulla_richiesta_in_corso": pren.tool_annulla_richiesta_in_corso,
    "consulta_prenotazioni": pren.tool_consulta_prenotazioni,
    "dati_personali_non_disponibili": pren.tool_dati_personali_non_disponibili,
}
TOOL_PRENOTAZIONE = set(TOOL_FUNZIONI) - {"get_menu", "get_info_ristorante", "mostra_mappa_ristorante",
                                           "rispondi_domanda_generale"}
# Per questi tool, se vanno a buon fine, la risposta al cliente è il "messaggio" del tool stesso
# (testo fisso, con date/orari/codici corretti): l'LLM sceglie il tool, ma non può riscrivere i fatti.
TOOL_RISPOSTA_FISSA = {"controlla_disponibilita", "proponi_prenotazione", "conferma_prenotazione",
                       "modifica_prenotazione", "annulla_prenotazione",
                       "annulla_richiesta_in_corso", "consulta_prenotazioni"}


def risposta_fissa(usati: list, risultati: list) -> Optional[str]:
    """Testo del tool di prenotazione eseguito per ultimo, se è andato a buon fine."""
    for nome, ris in zip(reversed(usati), reversed(risultati)):
        if nome not in TOOL_PRENOTAZIONE:
            continue
        if ris.get("bloccata"):            # conferma anticipata respinta: si guarda il tool prima
            continue
        return ris.get("messaggio") if (nome in TOOL_RISPOSTA_FISSA and ris.get("ok")) else None
    return None


def _tool(nome: str, descrizione: str, parametri: Optional[dict] = None, obbligatori=()) -> dict:
    return {"type": "function", "function": {
        "name": nome, "description": descrizione,
        "parameters": {"type": "object", "properties": parametri or {}, "required": list(obbligatori)}}}


_P_PERSONE = {"type": "integer", "description": "Numero di persone (1-12), SOLO se detto dal cliente"}
_P_DATA = {"type": "string", "description": "Giorno in formato AAAA-MM-GG, SOLO se detto dal cliente (usa il CALENDARIO del contesto)"}
_P_ORA = {"type": "string", "description": "Orario in formato HH:MM, 24 ore, SOLO se detto dal cliente"}
_P_CODICE = {"type": "string", "description": "Codice prenotazione di 5 cifre"}

# Gli schemi che l'LLM vede: sono queste descrizioni a fargli scegliere il tool giusto.
TOOLS = [
    _tool("get_menu", "Restituisce il menu con piatti e prezzi. Usalo per qualsiasi domanda su cibo, "
          "piatti, bevande, dolci, prezzi.",
          {"categoria": {"type": "string", "description": "Opzionale: Antipasto, Primo, Secondo, Dolce, Bevanda"}}),
    _tool("get_info_ristorante", "Restituisce orari di apertura, indirizzo e contatti del ristorante. "
          "Usalo per domande su quando è aperto, dove si trova, come contattarlo."),
    _tool("mostra_mappa_ristorante", "Mostra al cliente la mappa con la posizione DEL NOSTRO RISTORANTE. "
          "Usalo solo se chiede dove si trova il ristorante, come raggiungerlo o la mappa del locale. "
          "NON usarlo per altri luoghi (monumenti, città, altri locali)."),
    _tool("rispondi_domanda_generale", "Risponde a QUALSIASI domanda che non riguarda il ristorante: cultura "
          "generale, geografia, storia, arte, scienza, curiosità, consigli, 'dove si trova' un luogo che non è il "
          "nostro ristorante. Non ha parametri: risponde alla domanda appena fatta dal cliente."),
    _tool("controlla_disponibilita", "Controlla se c'è posto, SENZA prenotare. Usalo se il cliente chiede "
          "solo se un tavolo è libero.",
          {"persone": _P_PERSONE, "data": _P_DATA, "ora": _P_ORA}, ["persone", "data", "ora"]),
    _tool("proponi_prenotazione", "Passo 1 di una NUOVA prenotazione: valida i dati e prepara il riepilogo. "
          "NON prenota ancora. Chiamalo solo se il cliente ha detto TUTTI E TRE i dati (persone, giorno, ora): "
          "se ne manca uno, non chiamarlo e chiedilo al cliente. Richiamalo se il cliente cambia i dati prima di confermare.",
          {"persone": _P_PERSONE, "data": _P_DATA, "ora": _P_ORA}, ["persone", "data", "ora"]),
    _tool("conferma_prenotazione", "Passo 2: salva davvero la prenotazione (o la modifica) proposta. "
          "Chiamalo SOLO dopo che il cliente ha risposto di sì al riepilogo, mai nello stesso messaggio "
          "in cui lo hai proposto."),
    _tool("modifica_prenotazione", "Cambia una prenotazione già confermata. Serve il codice di 5 cifre. "
          "Passa solo i campi da cambiare. Poi serve la conferma del cliente.",
          {"codice": _P_CODICE, "persone": _P_PERSONE, "data": _P_DATA, "ora": _P_ORA}),
    _tool("annulla_prenotazione", "CANCELLA/ANNULLA/DISDICE una prenotazione già confermata (il cliente può dire "
          "anche 'appuntamento', 'tavolo', 'prenotazione'). Passa il codice di 5 cifre se il cliente lo ha detto o "
          "ha scelto una prenotazione; altrimenti OMETTI il codice (non scrivere null): il tool gli chiederà quale.",
          {"codice": _P_CODICE}),
    _tool("annulla_richiesta_in_corso", "SOLO se hai appena fatto un riepilogo NON ancora confermato e il cliente "
          "rinuncia a prenotare. NON per cancellare prenotazioni già fatte (usa annulla_prenotazione)."),
    _tool("consulta_prenotazioni", "Mostra i dettagli di una prenotazione dato il codice, oppure l'elenco "
          "delle prenotazioni fatte in questa conversazione. Per l'elenco OMETTI il codice (non scrivere null).",
          {"codice": _P_CODICE}),
    _tool("dati_personali_non_disponibili", "Da usare quando il cliente chiede di vedere, ripetere o riepilogare "
          "dati personali (nome, cognome, email, telefono) suoi o di altri clienti. NON per telefono/contatti "
          "del ristorante, NON per il riepilogo/elenco delle prenotazioni (usa consulta_prenotazioni)."),
]


def esegui_tool(nome: str, args, stato: dict) -> dict:
    """Esegue il tool richiesto dall'LLM. Qualunque errore diventa un risultato leggibile dall'LLM."""
    funzione = TOOL_FUNZIONI.get(nome)
    if funzione is None:
        return {"ok": False, "messaggio": f"Il tool '{nome}' non esiste."}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    try:
        return funzione(stato, **args)
    except TypeError as e:                         # argomenti inventati dall'LLM
        print(f"[TOOL] {nome} argomenti non validi {args}: {e}")
        return {"ok": False, "messaggio": "Parametri non validi per questo strumento."}
    except Exception as e:
        print(f"[TOOL] {nome} errore: {e}")
        return {"ok": False, "messaggio": "Errore interno, riprova tra poco."}


# LLM
def _calendario(oggi: datetime.date, giorni: int = 14) -> str:
    """L'LLM sbaglia spesso i calcoli di calendario: gli diamo i prossimi giorni già pronti."""
    righe = []
    for i in range(giorni):
        d = oggi + datetime.timedelta(days=i)
        chiuso = " (CHIUSO)" if d.weekday() in pren.GIORNI_CHIUSURA else ""
        etichetta = {0: "oggi", 1: "domani", 2: "dopodomani"}.get(i, "")
        righe.append(f"{pren.fmt_data(d)} = {d.isoformat()}{chiuso}" + (f" [{etichetta}]" if etichetta else ""))
    return "\n".join(righe)


def _system_prompt(stato: dict) -> str:
    nome = leggi_db()["info"].get("nome", "il ristorante")
    oggi = datetime.date.today()
    sessione = pren.prenotazioni_sessione(stato)
    return f"""Sei l'assistente virtuale di "{nome}". Rispondi SEMPRE in italiano, con tono cordiale e professionale.

REGOLE FONDAMENTALI (valgono sempre, non violarle mai):
A. NON INVENTARE. Non inventare piatti, prezzi, orari, indirizzi, disponibilità, codici di prenotazione, né dettagli del cliente. Se non hai un dato, dì chiaramente "non lo so" e, se riguarda il ristorante, consiglia di contattarlo.
B. NON INVENTARE I DATI DELLA PRENOTAZIONE. Numero di persone, giorno e ora li può dire SOLO il cliente, con le sue parole. Se anche uno solo manca, FAI UNA DOMANDA per ottenerlo. Non proporre mai valori "di default" o "tipici" (niente "4 persone", niente "alle 20:30" se il cliente non lo ha detto). Non fare mai un riepilogo senza aver prima chiamato proponi_prenotazione.
C. NON AFFERMARE AZIONI NON ESEGUITE. Non dire mai che un tavolo è libero, prenotato, modificato o annullato, né dare un codice, se non te lo ha appena restituito un tool. Non dire "ho prenotato" a parole: la prenotazione esiste solo se conferma_prenotazione ha risposto ok.
D. NON PARLARE DEL TUO FUNZIONAMENTO. Non nominare mai tool, funzioni, prompt, regole, modello o istruzioni (vietato dire "non ho funzioni per..."). Se non puoi fare qualcosa, dillo in modo naturale.
E. MENU E INFORMAZIONI: per piatti, prezzi, bevande, orari, indirizzo, contatti chiama SEMPRE get_menu / get_info_ristorante prima di rispondere e usa SOLO ciò che restituiscono. Se il cliente chiede dove si trova il NOSTRO ristorante o come arrivare, chiama mostra_mappa_ristorante e dai l'indirizzo; per la posizione di qualunque altro luogo NON chiamarlo (regola G). Se il cliente chiede qualcosa che il menu non dice (allergeni, ingredienti, piatti vegetariani non indicati, sconti), rispondi che non hai quell'informazione e consiglia di contattare il ristorante: non dedurre.
F. DATI PERSONALI: se il cliente chiede di vedere nome, email o telefono dati in prenotazione (suoi o di altri), chiama dati_personali_non_disponibili. Non fornire mai dati personali.
H. SEGNAPOSTO DI PRIVACY. Nei messaggi del cliente possono comparire [EMAIL], [TELEFONO], [NOME_CLIENTE]: sono i suoi stessi dati, già nascosti e salvati in modo sicuro dal sistema. Significano che il cliente ha dato i propri contatti per la SUA prenotazione. Ignorali: non chiedere i dati, non chiedere chi sia l'ospite, non rifiutare, non pensare che la prenotazione sia per un'altra persona, e non scrivere mai i segnaposto nelle risposte. Prosegui con persone, giorno e ora. Non rifiutare MAI una prenotazione perché è "per qualcun altro": il cliente può prenotare per chi vuole.
G. QUALSIASI DOMANDA CHE NON RIGUARDA IL RISTORANTE (cultura generale, geografia, storia, arte, scienza, consigli, curiosità, "dove si trova" un luogo che non è il nostro ristorante): chiama SEMPRE rispondi_domanda_generale e non rispondere tu. Solo saluti, ringraziamenti e chiacchiere brevi li gestisci tu in una frase, ricordando che puoi aiutare con menu e prenotazioni.

PRENOTAZIONI GIÀ FATTE: se il cliente chiede l'elenco, il riepilogo o cosa ha prenotato, chiama SEMPRE consulta_prenotazioni SENZA il parametro codice (oppure con il codice, se lo ha dato). Non rispondere mai a memoria che "non ha prenotazioni". Mai scrivere "null" o "none" nei parametri: se un parametro non c'è, omettilo.

COME PRENOTARE:
1. Raccogli numero di persone, giorno e ora (chiedi SOLO ciò che manca, una domanda alla volta se serve).
2. Quando hai tutti e tre i dati detti dal cliente, chiama proponi_prenotazione. Il riepilogo lo scrive il tool.
3. Solo dopo che il cliente risponde di sì, chiama conferma_prenotazione. Se invece cambia idea sui dati, richiama proponi_prenotazione con i dati nuovi; se rinuncia, chiama annulla_richiesta_in_corso.
4. CANCELLARE / ANNULLARE / DISDIRE una prenotazione già confermata (anche se il cliente dice "appuntamento", "tavolo" o "richiesta"): chiama annulla_prenotazione. Se il cliente non ha indicato quale, ometti il codice: il tool gli mostrerà le sue prenotazioni e gli chiederà quale. Non annullare mai a caso. Lo stesso vale per modifica_prenotazione. annulla_richiesta_in_corso è SOLO per chi rinuncia a un riepilogo non ancora confermato.
5. Se un tool risponde con un errore (orario chiuso, giorno di chiusura, posto non disponibile, dato mancante), spiega il motivo al cliente con parole tue e proponi un'alternativa o chiedi il dato mancante.
6. Per i giorni usa il CALENDARIO qui sotto, non calcolare le date a mente. Gli orari nei tool sono in formato 24 ore.

ESEMPI DI COMPORTAMENTO CORRETTO:
- Cliente: "voglio prenotare un tavolo" -> nessun tool, chiedi: "Volentieri! Per quante persone, per quale giorno e a che ora?"
- Cliente: "domani alle 20 siamo in 4" -> chiami proponi_prenotazione(persone=4, data=<data di domani dal CALENDARIO>, ora="20:00").
- Cliente: "vorrei prenotare con email [EMAIL] per 2 persone il 20 ottobre alle 13" -> chiami proponi_prenotazione(persone=2, data=<20 ottobre dal CALENDARIO>, ora="13:00"): il segnaposto si ignora.
- Cliente: "vorrei cancellare un appuntamento" -> chiami annulla_prenotazione SENZA codice (il tool mostra le prenotazioni e chiede quale). NON annulla_richiesta_in_corso.
- Cliente: "annulla la 12345" o, dopo l'elenco, "sì" / "la prima" -> chiami annulla_prenotazione con quel codice.
- Cliente: "fammi un riepilogo delle mie prenotazioni" -> chiami consulta_prenotazioni senza parametri.
- Cliente: "sabato sera" (mancano persone e ora) -> nessun tool, chiedi per quante persone e a che ora.
- Cliente (dopo il riepilogo): "sì, confermo" -> chiami conferma_prenotazione.
- Cliente: "avete piatti vegani?" -> chiami get_menu e rispondi solo in base al menu; se non è indicato, di' che non lo sai.
- Cliente: "dove si trova la Gioconda?" / "chi ha scritto la Divina Commedia?" -> chiami rispondi_domanda_generale (NON mostra_mappa_ristorante, NON rispondi tu).
- Cliente: "dove siete?" / "come vi raggiungo?" -> chiami mostra_mappa_ristorante e dai l'indirizzo.

FORMATO: massimo 3 frasi (tranne quando elenchi il menu), senza elenchi puntati, senza markdown e senza emoji: la risposta viene letta ad alta voce. Parla solo come assistente: non scrivere mai le battute del cliente.

CONTESTO:
Oggi è {pren.fmt_data(oggi)}.
CALENDARIO:
{_calendario(oggi)}
PRENOTAZIONI DI QUESTO CLIENTE: {' | '.join(sessione) if sessione else 'nessuna'}"""


def _chiama_ollama(messages: list, usa_tool: bool = True, modello: Optional[str] = None,
                   temperatura: float = 0.1) -> Optional[dict]:
    payload = {
        "model": modello or MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperatura, "top_p": 0.8, "num_predict": 300},  # creatività minima: meno invenzioni
    }
    if usa_tool:
        payload["tools"] = TOOLS
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=LLM_TIMEOUT)
        if r.status_code == 400 and "does not support tools" in r.text:
            print(f"[LLM] il modello '{MODEL_NAME}' non supporta i tool: usa es. 'ollama pull llama3.1' "
                  f"e MODEL_NAME=llama3.1")
            return None
        r.raise_for_status()
        return r.json().get("message", {})
    except Exception as e:
        print(f"[LLM] errore: {e}")
        return None


def _pulisci(testo: str) -> str:
    """Pulizia dell'OUTPUT (non serve a capire cosa vuole il cliente): via markdown e
    battute inventate del cliente, perché la risposta viene letta ad alta voce."""
    testo = re.sub(r"[*_`#]+", "", testo or "")
    return re.split(r"\n\s*(?:Cliente|Utente)\s*:", testo)[0].strip()


def chiedi_llm(messaggio: str, stato: dict):
    """Dialogo con l'LLM + tool. Ritorna (risposta|None, [nomi dei tool usati], ultimi risultati)."""
    stato["ultimo_messaggio"] = messaggio
    messages = ([{"role": "system", "content": _system_prompt(stato)}]
                + stato["storia"] + [{"role": "user", "content": messaggio}])
    usati, risultati = [], []
    for _ in range(MAX_GIRI_TOOL + 1):
        msg = _chiama_ollama(messages, usa_tool=len(usati) < MAX_GIRI_TOOL)
        if msg is None:
            return None, usati, risultati
        chiamate = msg.get("tool_calls") or []
        if not chiamate:
            testo = _pulisci(msg.get("content", ""))
            if not testo and risultati:               # l'LLM tace dopo un tool: usiamo il messaggio del tool
                testo = risultati[-1].get("messaggio", "")
            return (testo or None), usati, risultati
        messages.append(msg)
        for c in chiamate:
            f = c.get("function", {})
            nome, args = f.get("name", ""), f.get("arguments") or {}
            ris = esegui_tool(nome, args, stato)
            print(f"[TOOL] {nome}({args}) -> {ris.get('ok')}")
            usati.append(nome)
            risultati.append(ris)
            messages.append({"role": "tool", "tool_name": nome,
                             "content": json.dumps(ris, ensure_ascii=False, default=str)})
    return None, usati, risultati


def risposta_di_riserva() -> str:
    """Se l'LLM non è raggiungibile: niente regex, solo i dati fissi letti dal DB."""
    info = leggi_db()["info"]
    base = ("Scusa, in questo momento il mio assistente non è raggiungibile e non posso gestire "
            "le prenotazioni. ")
    dettagli = testo_info(info)
    return base + (("Intanto ecco le informazioni del ristorante.\n" + dettagli) if dettagli
                   else "Riprova tra poco.")


# API  Invia una richiesta HTTP POST all'endpoint /api/chat
@app.post("/api/chat")
def chat_endpoint(req: UserRequest):
    """Endpoint unico usato da CHAT e VOICE (def sincrona: FastAPI la esegue in un thread)."""
    stato = get_stato(req.session_id)
    stato["turno"] += 1
    if req.dati_cifrati and re.fullmatch(r"[A-Za-z0-9_=\-]{1,2000}", req.dati_cifrati):
        stato["dati_cifrati"] = req.dati_cifrati         # solo testo cifrato: mai loggato, mai letto

    risposta, usati, risultati = chiedi_llm(req.message, stato)   # l'LLM decide se usare i tool
    if "dati_personali_non_disponibili" in usati:
        fonte, risposta = "gdpr", pren.MSG_GDPR                    # testo fisso, non riformulato dall'LLM
    elif risposta is None:
        fonte, risposta = "riserva", risposta_di_riserva()
    else:
        fonte = "prenotazioni" if TOOL_PRENOTAZIONE & set(usati) else "llm"
        fissa = risposta_fissa(usati, risultati)
        diretta = next((r["messaggio"] for r in reversed(risultati)
                        if r.get("risposta_diretta") and r.get("ok")), None)
        if fissa:                                         # niente date/codici riscritti dall'LLM
            risposta = fissa
        elif diretta:                                     # risposta pulita di cultura generale
            risposta = diretta

    stato["storia"] += [{"role": "user", "content": req.message},
                        {"role": "assistant", "content": risposta}]
    stato["storia"] = stato["storia"][-pren.STORIA_MAX:]
    mappa = any(ris.get("mappa") for ris in risultati)           # il client mostra la mappa del ristorante
    return {"reply": risposta, "fonte": fonte, "mappa": mappa}


@app.get("/api/health")
def health():
    return {"status": "ok", "db": os.path.exists(DB_PATH), "modello": MODEL_NAME}


if __name__ == "__main__":  # Uvicorn gestisce l'esecuzione dell'applicazione FastAPI.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)       # 0.0.0.0: raggiungibile dalla LAN
