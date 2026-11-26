# Allenamento del modello loghi

Questa cartella è separata dal sito: serve ad allenare il rilevatore di loghi (YOLO) che il sito e la CLI usano come fase "Loghi". Il risultato è un file `models/logo.pt`. Senza questo file la fase loghi resta spenta; volti, soggetto e sfondo funzionano comunque.

## Come funziona

1. Raccogli foto con loghi e segna dove sono (etichette).
2. `prepare_dataset.py` controlla le etichette, divide in allenamento/validazione e scrive `data.yaml`.
3. `train_logo.py` allena, misura il risultato e installa il modello in `models/logo.pt`.
4. Riavvii il sito: in alto la voce "Loghi" diventa attiva.

## 1. Dataset

Serve il formato YOLO: per ogni immagine un file `.txt` con lo stesso nome, una riga per logo:

```text
0 0.52 0.31 0.18 0.09
```

I cinque numeri sono classe, centro x, centro y, larghezza, altezza, tutti tra 0 e 1 rispetto alla dimensione dell'immagine. Per oscurare i loghi basta una sola classe (`logo`).

Strade per averlo:

- **Annotare le tue foto** con Roboflow (gratis nel browser), CVAT, Label Studio o LabelImg, ed esportare in formato "YOLO".
- **Usare OpenLogo o LogoDet-3K**: hanno annotazioni Pascal VOC (XML). Convertile con `python training/voc_to_yolo.py --source cartella_scaricata --dest training/source` (unisce tutti i marchi in una classe `logo`; con `--keep-classes` li mantiene separati).
- **Esempi negativi**: aggiungi foto senza loghi (file `.txt` vuoto o assente). Riducono i falsi allarmi.

Per un primo risultato servono almeno 200 immagini con loghi; per qualcosa di affidabile qualche migliaio, con sfondi, luci e dimensioni diverse.

Non mettere nel repository foto personali: la cartella `training/source`, `training/dataset` e `training/runs` è già esclusa da `.gitignore`.

## 2. Preparare

```bash
python training/prepare_dataset.py --source training/source
```

Opzioni utili: `--val-ratio 0.2` (quota per la validazione), `--classes logo`, `--force` per rifare il dataset. Se un'etichetta è sbagliata (classe inesistente, coordinate fuori da 0-1, righe incomplete) lo script lo dice e non copia nulla.

## 3. Allenare

```bash
python training/train_logo.py --epochs 50
```

- Parte da `yolov8n.pt`, il modello più piccolo; la prima volta lo scarica da internet.
- Su CPU (Codespaces, portatile) è lento: decine di minuti per epoca con migliaia di immagini. Per usare una GPU gratuita apri `train_colab.ipynb` su Google Colab.
- Se finisce la memoria abbassa `--batch` (per esempio 4) o `--imgsz`.
- Alla fine stampa precisione, richiamo e mAP sulla validazione, copia il modello migliore in `models/logo.pt` (tenendo una copia di quello precedente) e scrive lo SHA-256 in `models/models.lock.json`. Con `--no-install` si ferma prima della copia.

Come leggere i numeri: il **richiamo** dice quanti loghi veri vengono trovati (per la privacy è quello che conta di più); la **precisione** dice quanti riquadri sono davvero loghi. Un mAP50 sotto 0.5 vuol dire che servono più dati o più epoche.

## 4. Controllare a occhio

```bash
python training/evaluate_model.py --preview cartella_con_foto_di_prova
```

Salva in `training/runs/preview` le foto con i riquadri trovati. Usa foto che il modello non ha mai visto. Con `--data training/dataset/data.yaml` ricalcola le metriche.

## 5. Usare il modello

Il file `models/logo.pt` viene letto dal sito (`python serve.py`) e dalla CLI (`python run.py --logo-model models/logo.pt ...`). Per il sito pubblicato online il file deve entrare nell'immagine Docker: vedi `docs/DEPLOY.md`, sezione "Modello dei loghi". Il file non finisce su GitHub (`*.pt` è in `.gitignore`): in Codespaces vive solo in quel Codespace, quindi scaricalo e conservalo se vuoi riusarlo.

I file `.pt` contengono codice eseguibile (pickle): carica solo modelli allenati da te o di cui ti fidi. Il controllo dello SHA-256 serve proprio a questo.

## Limiti

- Un modello allenato su pochi marchi non riconosce quelli che non ha visto.
- Loghi molto piccoli o parziali sfuggono: la fase loghi riduce il rischio, non lo azzera.
- L'allenamento non migliora volti, soggetto e sfondo: quelli usano modelli già pronti.
