import inspect
import io
import json
import os
import re
import sys
import types
import wave
import uuid
import warnings
from datetime import date, time
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/chat")   # IP del BE (macchina 1) sulla LAN

NOME_RISTORANTE = "Il Bocconcino osteria romana"
INDIRIZZO = "Via Ostilia, 23, 00184 Roma RM"
SITO_WEB = "https://ilbocconcino.com"
TELEFONO = ""                                   
LAT, LON = 41.8887, 12.4982                      
ORARI = "Martedì–Domenica 12:00–15:00 / 19:00–23:00 (Lunedì chiuso)"   
GIORNO_CHIUSURA = 0                              #lunedi

GIORNI = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
MESI = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
        "agosto", "settembre", "ottobre", "novembre", "dicembre"]

# Frasi che Whisper "inventa" quando sente solo silenzio
ALLUCINAZIONI = ("sottotitoli", "grazie per la visione",
                 "iscriviti", "buona visione", "alla prossima")

# Da Streamlit 1.47 la chat_input ha il microfono integrato 
AUDIO_NEL_CHAT = "accept_audio" in inspect.signature(st.chat_input).parameters


# ============================================================
# PRIVACY: filtro pre-invio al backend
# ============================================================
def anonimizza_ed_estrai(testo: str):
    """Rimuove i dati personali PRIMA dell'invio al BE (vincolo 3 della traccia) e li
    restituisce a parte, cosi' il CLIENT puo' cifrarli in locale.
    Ritorna (testo_anonimo, dati) con dati = {"email": ..., "telefono": ..., "nome": ...}."""
    dati = {}

    def _email(m):
        dati["email"] = m.group(0)
        return "[EMAIL]"

    def _telefono(m):
        dati["telefono"] = re.sub(r"[\s.\-]", "", m.group(0))
        return "[TELEFONO]"

    def _nome(m):
        dati["nome"] = m.group(2)
        return f"{m.group(1)} [NOME_CLIENTE]"

    # 1. Email
    testo = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', _email, testo)
    # 2. Telefoni: 8-11 cifre, anche separate da spazi/punti/trattini (il codice prenotazione, 5 cifre, passa)
    testo = re.sub(r'(?<!\d)(?:\+39[\s-]*)?\d(?:[\s.\-]?\d){7,10}(?!\d)', _telefono, testo)
    # 3. Nomi dopo "mi chiamo / a nome (di) / il mio nome e'": qualunque parola, anche minuscola
    #    (Whisper e chi scrive di fretta non usano le maiuscole); la 2a parola solo se maiuscola
    testo = re.sub(r'\b((?i:mi chiamo|a nome(?: di)?|il mio nome è))\s+([^\W\d_]+(?:\s+[A-ZÀ-Ù][a-zà-ù]+)?)',
                   _nome, testo)
    # 4. "sono Mario Rossi": qui il nome deve avere la maiuscola (cosi' "sono due persone" non viene toccato)
    testo = re.sub(r'\b((?i:sono))\s+([A-ZÀ-Ù][a-zà-ù]+(?:\s+[A-ZÀ-Ù][a-zà-ù]+)?)', _nome, testo)
    return testo, dati


def anonimizza_dati(testo: str) -> str:
    """Solo il testo anonimo (stessa funzione di prima)."""
    return anonimizza_ed_estrai(testo)[0]


# ---------------------------------------------------------------- CIFRATURA ---
# I dati personali vengono cifrati QUI, sul client, con una chiave che non lascia mai
# questo dispositivo: il BE riceve e salva nel DB solo il testo cifrato e non puo' leggerlo.
def _chiave_cifratura() -> bytes:
    chiave = os.getenv("CHIAVE_CIFRATURA")           # opzionale: stessa chiave su piu' dispositivi
    if chiave:
        return chiave.encode()
    percorso = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chiave_cifratura")
    if os.path.exists(percorso):
        with open(percorso, "rb") as f:
            return f.read().strip()
    from cryptography.fernet import Fernet
    chiave = Fernet.generate_key()
    with open(percorso, "wb") as f:
        f.write(chiave)
    try:
        os.chmod(percorso, 0o600)
    except OSError:
        pass
    return chiave


def cifra_dati(dati: dict):
    """Cifra i dati personali (Fernet: AES-128 + HMAC). None se non ce ne sono."""
    if not dati:
        return None
    try:
        from cryptography.fernet import Fernet
        grezzo = json.dumps(dati, ensure_ascii=False).encode("utf-8")
        return Fernet(_chiave_cifratura()).encrypt(grezzo).decode("ascii")
    except ImportError:
        print("[PRIVACY] manca il pacchetto 'cryptography' (pip install cryptography): "
              "i dati personali NON vengono salvati")
        return None


# ============================================================
# VOCE: trascrizione (Whisper) e sintesi (voce del browser)
# ============================================================
def _neutralizza_av():
    """faster-whisper importa 'av' (PyAV) solo per aprire file audio, che qui non usiamo
    (passiamo array numpy). Se il criterio di sicurezza blocca la sua DLL, lo sostituiamo
    con un modulo vuoto così l'import di faster-whisper non fallisce."""
    try:
        import av  # noqa: F401
        return
    except Exception:
        pass
    for nome in [k for k in sys.modules if k == "av" or k.startswith("av.")]:
        del sys.modules[nome]

    class _Vuoto:
        def __getattr__(self, nome):
            if nome.startswith("__"):
                raise AttributeError(nome)
            return _Vuoto()

        def __call__(self, *a, **k):
            raise RuntimeError("PyAV (av) non è disponibile su questo PC")

    finto = types.ModuleType("av")
    finto.__getattr__ = lambda nome: (_ for _ in ()).throw(AttributeError(nome)) \
        if nome.startswith("__") else _Vuoto()
    sys.modules["av"] = finto


def _vad_disponibile() -> bool:
    """Il filtro silenzi (VAD) usa onnxruntime: se è bloccato lo spegniamo."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def get_whisper():
    # Cache: il modello viene caricato una sola volta (niente PyTorch, solo CPU)
    _neutralizza_av()
    from faster_whisper import WhisperModel
    return WhisperModel("small", device="cpu", compute_type="int8")


def _wav_a_float(audio_bytes: bytes) -> np.ndarray:
    """WAV registrato dal browser -> array float32 mono a 16 kHz."""
    with wave.open(io.BytesIO(audio_bytes)) as w:
        rate, canali, larghezza = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if larghezza == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif larghezza == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif larghezza == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"WAV a {larghezza * 8} bit non supportato")
    if canali > 1:
        audio = audio[: len(audio) // canali * canali].reshape(-1, canali).mean(axis=1)
    if rate != 16000 and len(audio) > 1:
        n = int(len(audio) * 16000 / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def trascrivi(audio_bytes: bytes) -> str:
    """Trascrive l'audio registrato dal microfono del browser."""
    try:
        audio = _wav_a_float(audio_bytes)
        if len(audio) < 1600:                       # < 0,1 secondi
            return ""
        picco = float(np.max(np.abs(audio)))
        if picco > 0:
            audio = audio / picco * 0.9             # alza le voci troppo basse
        segmenti, _ = get_whisper().transcribe(
            audio, language="it", beam_size=5, temperature=0.0, vad_filter=_vad_disponibile(),
            condition_on_previous_text=False,
            initial_prompt=(f"Conversazione con {NOME_RISTORANTE}. Prenotazione di un tavolo per "
                            "stasera, domani, venerdì, sabato, per due, tre, quattro persone. "
                            "Menu, carbonara, cacio e pepe, carciofi, tiramisù, orari, indirizzo."))
        parti = [x.text for x in segmenti if x.no_speech_prob < 0.6 and x.avg_logprob > -1.2]
        return " ".join(parti).strip()
    except ImportError as e:
        st.error(f"Impossibile caricare faster-whisper. Dettaglio: {e}")
    except Exception as e:
        st.error(f"Errore nella trascrizione: {e}")
    return ""


def e_allucinazione(testo: str) -> bool:
    t = testo.lower()
    if len(t) <= 2:
        return True
    # Solo sulle frasi brevi, così "grazie mille, prenoto per due" passa
    return len(t) < 40 and any(a in t for a in ALLUCINAZIONI)


def parla_nel_browser(testo: str):
    """Legge il testo con la sintesi vocale del browser."""
    pulito = re.sub(r'[*_`#]', '', testo).replace('€', ' euro')
    pulito = re.sub(r'\b(\d)(\d)(\d)(\d)(\d)\b', r'\1 \2 \3 \4 \5', pulito)   # i codici si leggono cifra per cifra
    components.html(
        f"""<script>
        const u = new SpeechSynthesisUtterance({json.dumps(pulito)});
        u.lang = 'it-IT'; u.rate = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
        </script>""",
        height=0,
    )


# ============================================================
# BACKEND
# ============================================================
def chiedi_al_backend(testo: str):
    """Ritorna (risposta, mappa): 'mappa' è True se il BE (cioè l'LLM) vuole mostrare la mappa del ristorante."""
    testo_anonimo, nuovi = anonimizza_ed_estrai(testo)
    if nuovi:                                        # nome/email/telefono: cifrati QUI, in locale
        st.session_state.dati_personali.update(nuovi)
        st.session_state.dati_cifrati = cifra_dati(st.session_state.dati_personali)
    payload = {"message": testo_anonimo, "session_id": st.session_state.session_id}
    if st.session_state.dati_cifrati:
        payload["dati_cifrati"] = st.session_state.dati_cifrati   # solo testo cifrato
    try:
        res = requests.post(BACKEND_URL, json=payload, timeout=120)
        if res.status_code == 200:
            dati = res.json()
            return dati.get("reply", ""), bool(dati.get("mappa", False))
        return "⚠️ Errore di comunicazione col server (Backend).", False
    except requests.exceptions.ConnectionError:
        return "⚠️ Errore di connessione: il Backend è acceso e raggiungibile in rete?", False
    except requests.exceptions.Timeout:
        return "⚠️ Il server ci sta mettendo troppo a rispondere. Riprova.", False


def mostra_mappa(chiave: str):
    """Mappa + pulsanti. 'chiave' deve essere diversa a ogni chiamata (evita errori di ID duplicati)."""
    st.map(pd.DataFrame({"lat": [LAT], "lon": [LON]}), zoom=16, height=250)

    # Il link usa nome + indirizzo, così Google Maps apre la scheda vera del locale
    link_maps = ("https://www.google.com/maps/search/?api=1&query="
                 + quote_plus(f"{NOME_RISTORANTE} {INDIRIZZO}"))
    bottoni = [("🗺️ Apri in Google Maps", link_maps), ("🌐 Sito web", SITO_WEB)]
    if TELEFONO:
        bottoni.append(("📞 Chiama", f"tel:{TELEFONO}"))

    for col, (etichetta, url), i in zip(st.columns(len(bottoni)), bottoni, range(len(bottoni))):
        col.link_button(etichetta, url, use_container_width=True, key=f"{chiave}_{i}")


# ============================================================
# INTERFACCIA
# ============================================================
st.set_page_config(page_title=NOME_RISTORANTE, page_icon="🍝", layout="centered")
st.title(f"{NOME_RISTORANTE} 🍝")
st.write("Benvenuto! Chiedimi il menu, gli orari o prenota un tavolo: **scrivendo o parlando**.")
if AUDIO_NEL_CHAT:
    st.caption("🎤 Per parlare, premi il microfono nella barra in basso.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ultimo_audio" not in st.session_state:
    st.session_state.ultimo_audio = None
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex   # il BE ricorda la prenotazione per questa sessione
if "dati_personali" not in st.session_state:
    st.session_state.dati_personali = {}             # restano SOLO nel browser/PC del client
    st.session_state.dati_cifrati = None

pending = None        
da_voce = False


with st.sidebar:
    st.header("ℹ️ Info")
    st.write(f"📍 {INDIRIZZO}")
    st.write(f"🕐 {ORARI}")
    mostra_mappa("sidebar")

    st.divider()
    st.header("📅 Prenota veloce")
    with st.form("form_prenota"):
        data = st.date_input("Data", min_value=date.today())
        ora = st.time_input("Ora", value=time(20, 0), step=1800)
        persone = st.number_input("Persone", min_value=1, max_value=20, value=2)
        invia = st.form_submit_button("Richiedi prenotazione", use_container_width=True)
    if invia:
        if data.weekday() == GIORNO_CHIUSURA:
            st.warning(f"Il {GIORNI[GIORNO_CHIUSURA]} siamo chiusi, scegli un altro giorno 🙂")
        else:
            pending = (f"Vorrei prenotare un tavolo per {persone} persone "
                       f"il {data.day} {MESI[data.month - 1]} {data.year} alle {ora.strftime('%H:%M')}.")

    st.divider()
    voce_attiva = st.toggle("🔊 Risposta vocale", value=True)
    if st.button("🗑️ Nuova conversazione", use_container_width=True):
        st.session_state.messages = []
        st.session_state.ultimo_audio = None
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.dati_personali = {}
        st.session_state.dati_cifrati = None
        st.rerun()

# ---------- Azioni rapide ----------
st.caption("Azioni rapide")
AZIONI = {
    "📜 Menu": "Mostrami il menu",
    "🕐 Orari": "Quali sono gli orari di apertura?",
    "📍 Dove siete?": "Dove si trova il ristorante?",
}
for col, (etichetta, domanda) in zip(st.columns(len(AZIONI)), AZIONI.items()):
    if col.button(etichetta, use_container_width=True):
        pending = domanda

# ---------- Microfono di riserva (solo con Streamlit < 1.47) ----------
if not AUDIO_NEL_CHAT:
    audio = st.audio_input("🎤 Clicca qui per parlare", sample_rate=16000)
    if audio is not None:
        audio_bytes = audio.getvalue()
        if audio_bytes != st.session_state.ultimo_audio:   # evita di rielaborare lo stesso audio
            st.session_state.ultimo_audio = audio_bytes
            with st.spinner("⏳ Trascrizione in corso (Whisper)..."):
                testo_voce = trascrivi(audio_bytes)
            if e_allucinazione(testo_voce):
                st.warning("Scusa, non ho capito. Puoi ripetere?")
            else:
                pending, da_voce = testo_voce, True

# ---------- Storico chat ----------
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("mappa"):
            mostra_mappa(f"storico_{i}")

# ---------- Barra di input (testo + microfono integrato) ----------
if AUDIO_NEL_CHAT:
    valore = st.chat_input("Scrivi qui o premi 🎤 per parlare...", accept_audio=True, audio_sample_rate=16000)
    if valore:
        testo_in, audio_in = valore.text, valore.audio
        if testo_in:
            pending = pending or testo_in
        elif audio_in is not None:
            with st.spinner("⏳ Trascrizione in corso (Whisper)..."):
                testo_voce = trascrivi(audio_in.getvalue())
            if e_allucinazione(testo_voce):
                st.warning("Scusa, non ho capito. Puoi ripetere?")
            else:
                pending, da_voce = testo_voce, True
else:
    if prompt := st.chat_input("Scrivi qui la tua richiesta..."):
        pending = pending or prompt

# ---------- Elaborazione del messaggio ----------
if pending:
    st.session_state.messages.append({"role": "user", "content": ("🎤 " if da_voce else "") + pending})
    with st.chat_message("user"):
        st.markdown(("🎤 " if da_voce else "") + pending)

    with st.chat_message("assistant"):
        with st.spinner("Sto pensando..."):
            risposta, vuole_mappa = chiedi_al_backend(pending)
        st.markdown(risposta)

        if vuole_mappa:                                   # deciso dall'LLM tramite il tool, non da parole chiave
            mostra_mappa(f"nuovo_{len(st.session_state.messages)}")

        if voce_attiva and not risposta.startswith("⚠️"):
            parla_nel_browser(risposta)

    st.session_state.messages.append({"role": "assistant", "content": risposta, "mappa": vuole_mappa})
