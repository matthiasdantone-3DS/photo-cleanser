# Photo Cleanser

Applicazione web che ripulisce le foto: l'utente apre il sito, carica un'immagine e riceve subito la versione con volti estranei, loghi e sfondo oscurati di nero e senza metadati (EXIF, XMP, IPTC, C2PA). Il soggetto principale resta com'è. Funziona con modelli di Machine Learning già pronti (YuNet per i volti, MediaPipe per soggetto e sfondo) e con un rilevatore di loghi che alleni tu.

Chi usa il sito non installa né configura nulla. Tu pubblichi il sito una volta sola (vedi sotto).

## Struttura

```text
photo-cleanser/
├── webapp/            sito web: backend FastAPI e pagina HTML/CSS/JS
├── redactor/          motore di oscuramento (usato da sito e CLI)
├── training/          addestramento del modello dei loghi (separato dal sito)
├── models/            download e controllo dei modelli (checksum)
├── docs/              DEPLOY.md (pubblicazione), SICUREZZA.md (privacy e sicurezza)
├── Dockerfile         immagine pronta: sito + modelli
├── docker-compose.yml sito + HTTPS automatico (Caddy)
├── deploy/Caddyfile   configurazione del proxy HTTPS
├── fly.toml           configurazione per Fly.io
├── .github/workflows/ test e pubblicazione dell'immagine
├── scripts/           smoke test del servizio in esecuzione
├── tests/             test automatici (senza scaricare modelli)
├── run.py             programma da riga di comando
├── serve.py           avvia il sito in locale
└── setup.sh           installazione per lo sviluppo
```

## Pubblicare il sito

Su un server con Docker e un dominio:

```bash
git clone https://github.com/TUO_UTENTE/photo-cleanser.git
cd photo-cleanser
printf 'SITE_ADDRESS=redactor.tuodominio.it\n' > .env
docker compose up -d --build
```

Caddy attiva HTTPS da solo. Altre strade (Fly.io, immagine su GHCR, modello dei loghi, checksum, variabili): `docs/DEPLOY.md`. Come vengono protette le foto: `docs/SICUREZZA.md`.

## Sviluppo

### GitHub Codespaces

1. Code > Codespaces > Create codespace; aspetta la fine di `setup.sh`.
2. `python serve.py` e apri la porta 8000 dal pannello Ports (lasciala privata).

### In locale

```bash
python3.11 -m venv .venv && source .venv/bin/activate
bash setup.sh
python serve.py
```

Poi http://127.0.0.1:8000.

`setup.sh` scarica YuNet da OpenCV Zoo con `--allow-unverified` e registra il suo SHA-256 in `models/models.lock.json`. Verificalo sulla fonte e committa il lock.

### Test

```bash
python -m pytest
```

La CI di GitHub li esegue a ogni push.

## Allenare il modello dei loghi

La cartella `training/` è separata dal sito: dataset, conversione, allenamento (anche su Google Colab con GPU), valutazione. Il risultato è `models/logo.pt`, che il sito carica da solo. Guida: `training/README.md`. Per portarlo online: `docs/DEPLOY.md`, sezione "Modello dei loghi".

## Riga di comando

```bash
python run.py --input ./in --output ./out --report report.json
python run.py --input ./in --output ./out --logo-model models/logo.pt --dry-run
```

`--dry-run` mostra cosa verrebbe oscurato senza scrivere immagini. Soglie: `--face-score`, `--logo-conf`, `--mask-threshold`, `--pad-px`. I file esistenti in output non vengono mai sovrascritti.

## Limiti noti

- Soggetti multipli o parzialmente visibili: restano visibili tutte le componenti della maschera con area di almeno il 2% della più grande.
- Sfondo complesso e segmentazione imperfetta su capelli e contorni.
- Volti di profilo, piccoli o molto inclinati possono sfuggire a YuNet.
- Se non viene trovato nessun soggetto il file esce senza metadati ma NON oscurato (stato `nessun_soggetto`).
- Se un rilevatore non è disponibile il file ha stato `da_rivedere`; se la segmentazione non è disponibile va in `errore`.
- MediaPipe: dalla 0.10.30 non esiste più `mp.solutions`, per questo `requirements.txt` limita la versione sotto la 0.10.30.
- Più limiti e rischi residui: `docs/SICUREZZA.md`.
