# Componenti di terze parti

Il sito pubblicato include questi componenti, serviti dal sito stesso (nessuna CDN):

| Componente | Uso | Licenza |
| --- | --- | --- |
| MediaPipe Selfie Segmentation (`selfie_segmentation.tflite`) | segmentazione di soggetto e sfondo | Apache-2.0 |
| LiteRT.js (`@litertjs/core`) | esegue il modello TFLite nel browser | Apache-2.0 |
| YuNet 2023mar (OpenCV Zoo, di Shiqi Yu e collaboratori) | rilevamento dei volti | MIT |
| ONNX Runtime Web (`onnxruntime-web`) | esegue i modelli ONNX nel browser | MIT |
| YOLO / ultralytics (solo se alleni un modello di loghi) | addestramento del rilevatore di loghi | AGPL-3.0 |

Il modello di loghi, se presente, è addestrato con ultralytics: controlla le condizioni della sua licenza (AGPL-3.0) prima di pubblicarlo.
