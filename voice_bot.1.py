"""
Flusso voice bot:  microfono -> Whisper (locale) -> filtro privacy -> POST /api/chat
         (backend.py) 

Variabili d'ambiente opzionali:
  BACKEND_URL    es. http://192.168.1.10:8000/api/chat
  WHISPER_MODEL  tiny | base | small (default) | medium
  MIC_DEVICE     indice del microfono (vedi --devices)
"""
import collections
import json
import os
import platform
import re
import shutil
import subprocess
import sys 
import time
import uuid

import numpy as np
import requests

# --------------------------------------------------------------- CONFIG -----
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/chat")
SESSION_ID = uuid.uuid4().hex                 # una conversazione = un id (il BE ricorda la prenotazione)
BACKEND_TIMEOUT = 120                      
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
MIC_DEVICE = int(os.getenv("MIC_DEVICE")) if os.getenv("MIC_DEVICE") else None

SAMPLE_RATE = 16000
BLOCK_SEC = 0.03
BLOCK = int(SAMPLE_RATE * BLOCK_SEC)

WHISPER_PROMPT = (
    "Conversazione con il Ristorante Aroma. Prenotazione di un tavolo per "
    "stasera, domani, sabato, domenica, per due, tre, quattro persone. "
    "Menu: bruschetta, tagliere di salumi e formaggi, carbonara, "
    "risotto ai funghi porcini, bistecca fiorentina, tiramisù. "
    "Orari di apertura, pranzo, cena, indirizzo."
)
ALLUCINAZIONI = ("sottotitoli", "amara.org", "grazie per la visione",
                 "iscriviti", "buona visione", "alla prossima")


# VOCE (TTS) 
def _pulisci_per_voce(testo: str) -> str:
    testo = testo.replace("€", " euro")
    testo = re.sub(r"\b(\d)(\d)(\d)(\d)(\d)\b", r"\1 \2 \3 \4 \5", testo)   # codici cifra per cifra
    testo = re.sub(r"[*_`#>]+", "", testo)
    testo = re.sub(r"[^\w\s.,;:!?'\"()\-/àèéìòùÀÈÉÌÒÙ]", "", testo)
    return re.sub(r"\s+", " ", testo).strip()


def _tts_macos(testo: str) -> bool:
    if platform.system() != "Darwin" or not shutil.which("say"):
        return False
    for voce in ("Alice", "Luca"):
        if subprocess.run(["say", "-v", voce, testo]).returncode == 0:
            return True
    return subprocess.run(["say", testo]).returncode == 0


def _tts_pyttsx3(testo: str) -> bool:
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        for v in engine.getProperty("voices"):
            info = (v.name + " " + v.id).lower()
            langs = str(getattr(v, "languages", "")).lower()
            if "ital" in info or "it_" in langs or "it-" in langs or "'it'" in langs:
                engine.setProperty("voice", v.id)
                break
        engine.say(testo)
        engine.runAndWait()
        engine.stop()
        return True
    except Exception as e:
        print(f"[TTS pyttsx3 non disponibile: {e}]")
        return False


def _tts_espeak(testo: str) -> bool:
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            return subprocess.run([exe, "-v", "it", "-s", "150", testo]).returncode == 0
    return False


def parla(testo: str):
    """Stampa e pronuncia SEMPRE il testo (con 3 metodi di riserva)."""
    print(f"\n🤖 Ristorante-Bot: {testo}\n")
    voce = _pulisci_per_voce(testo)
    if not voce:
        return
    if not (_tts_macos(voce) or _tts_pyttsx3(voce) or _tts_espeak(voce)):
        print("[ATTENZIONE: nessun motore vocale funzionante]")
    time.sleep(0.4)  # lascia spegnere l'eco prima di riaprire il microfono


# PRIVACY 
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


# CIFRATURA
# I dati personali vengono cifrati QUI, sul client, con una chiave che non lascia mai
# questo dispositivo: il BE riceve e salva nel DB solo il testo cifrato e non puo' leggerlo.
def _chiave_cifratura() -> bytes:
    chiave = os.getenv("CHIAVE_CIFRATURA")          
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


_DATI_PERSONALI = {}      # nome/email/telefono di questa conversazione (restano SOLO qui, in chiaro)
_DATI_CIFRATI = None      # la loro versione cifrata, l'unica che va al BE


def _registra_dati_personali(nuovi: dict):
    global _DATI_CIFRATI
    if nuovi:
        _DATI_PERSONALI.update(nuovi)
        _DATI_CIFRATI = cifra_dati(_DATI_PERSONALI)


# BACKEND 
def chiedi_al_backend(testo: str) -> str:
    payload = {"message": testo, "session_id": SESSION_ID}
    if _DATI_CIFRATI:
        payload["dati_cifrati"] = _DATI_CIFRATI      # solo testo cifrato, il BE non ha la chiave
    try:
        r = requests.post(BACKEND_URL, json=payload, timeout=BACKEND_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return "Errore di rete. Verifica che il server centrale sia acceso."
    except requests.exceptions.Timeout:
        return "Il server sta impiegando troppo tempo a rispondere. Riprova tra poco."
    except Exception as e:
        print(f"[Errore richiesta: {e}]")
        return "Scusa, ho avuto un problema di comunicazione con il server."

    if r.status_code != 200:
        print(f"[Backend HTTP {r.status_code}: {r.text[:200]}]")
        return "Scusa, ho riscontrato un errore nel server centrale."
    try:
        reply = str(r.json().get("reply", "")).strip()
    except Exception:
        reply = ""
    return reply or "Scusa, non ho una risposta. Puoi riformulare la domanda?"


# MICROFONO + STT
def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) + 1e-9


def _stream():
    import sounddevice as sd
    return sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                          blocksize=BLOCK, device=MIC_DEVICE)


def calibra() -> float:
    """Misura il rumore ambiente e restituisce la soglia di voce."""
    with _stream() as s:
        livelli = [_rms(s.read(BLOCK)[0][:, 0]) for _ in range(int(1.5 / BLOCK_SEC))]
    ambiente = float(np.percentile(livelli, 90))
    soglia = max(ambiente * 3.0, 0.012)
    print(f"Rumore ambiente: {ambiente:.4f} -> soglia voce: {soglia:.4f}")
    return soglia


def ascolta(soglia: float, attesa_max=10.0, silenzio_fine=1.0, durata_max=20.0):
    """Aspetta che l'utente parli e registra fino a una pausa. None se silenzio."""
    pre = collections.deque(maxlen=10)      # ~300 ms prima dell'inizio voce
    frames, parlando, consecutivi, silenzio = [], False, 0, 0.0
    t0 = time.time()
    t_inizio = t0
    with _stream() as s:
        while True:
            blocco = s.read(BLOCK)[0][:, 0].copy()
            livello = _rms(blocco)
            if not parlando:
                pre.append(blocco)
                consecutivi = consecutivi + 1 if livello > soglia else 0
                if consecutivi >= 3:
                    parlando, frames, t_inizio = True, list(pre), time.time()
                elif time.time() - t0 > attesa_max:
                    return None
            else:
                frames.append(blocco)
                silenzio = silenzio + BLOCK_SEC if livello < soglia * 0.7 else 0.0
                if silenzio >= silenzio_fine or time.time() - t_inizio > durata_max:
                    break
    audio = np.concatenate(frames)
    return audio if len(audio) / SAMPLE_RATE > 0.4 else None


def carica_whisper():
    from faster_whisper import WhisperModel
    print(f"Caricamento Whisper '{WHISPER_MODEL}' (al primo avvio scarica il modello)...")
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def trascrivi(model, audio: np.ndarray) -> str:
    picco = float(np.max(np.abs(audio)))
    if picco > 0:
        audio = audio / picco * 0.9          # normalizza voci troppo basse
    segmenti, _ = model.transcribe(
        audio, language="it", beam_size=5, temperature=0.0,
        vad_filter=True, initial_prompt=WHISPER_PROMPT,
        condition_on_previous_text=False)
    parti = [s.text for s in segmenti
             if s.no_speech_prob < 0.6 and s.avg_logprob > -1.2]
    testo = " ".join(parti).strip()
    if any(a in testo.lower() for a in ALLUCINAZIONI):
        return ""
    return testo


#main
def _comando_uscita(testo: str) -> bool:
    return bool(re.search(r"\b(esci|spegni|arrivederci)\b", testo.lower()))


def _gestisci(testo_utente: str):
    print(f"👤 Hai detto: {testo_utente}")
    testo_pulito, nuovi = anonimizza_ed_estrai(testo_utente)
    _registra_dati_personali(nuovi)
    if testo_pulito != testo_utente:
        print(f"   (inviato al server: {testo_pulito})")
    if nuovi:
        print("   (dati personali cifrati in locale: il server non puo' leggerli)")
    parla(chiedi_al_backend(testo_pulito))


def run_text():
    parla("Modalità testo attiva. Scrivi 'esci' per terminare.")
    while True:
        try:
            t = input("\n👤 Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not t:
            continue
        if _comando_uscita(t):
            parla("Arrivederci!")
            break
        _gestisci(t)


def run_voice():
    try:
        model = carica_whisper()
    except Exception as e:
        print(f"ERRORE caricamento Whisper: {e}\n"
              "Installa con: pip install faster-whisper")
        sys.exit(1)

    parla("Sistema vocale attivato. Come posso aiutarti con la prenotazione?")

    try:
        soglia = calibra()
    except Exception as e:
        print(f"ERRORE microfono: {e}\n"
              "Controlla i permessi del microfono e prova: python voice_bot.py --devices")
        sys.exit(1)

    while True:
        try:
            print("\n🎤 In ascolto... (parla ora)")
            audio = ascolta(soglia)
            if audio is None:
                continue                      # silenzio: riascolta
            print("Elaborazione audio...")
            testo = trascrivi(model, audio)
            if len(testo) < 2:
                parla("Scusa, non ho capito bene. Puoi ripetere?")
                continue
            if _comando_uscita(testo):
                parla("Spegnimento del sistema vocale. Arrivederci!")
                break
            _gestisci(testo)
        except KeyboardInterrupt:
            parla("Spegnimento del sistema vocale. Arrivederci!")
            break
        except Exception as e:
            print(f"Errore di sistema: {e}")
            parla("Scusa, c'è stato un problema tecnico. Puoi ripetere?")


if __name__ == "__main__":
    if "--devices" in sys.argv:
        import sounddevice as sd
        print(sd.query_devices())
    elif "--text" in sys.argv:
        run_text()
    else:
        run_voice()
