# 🧠 MRI Tumor Classifier - TensorRT on Jetson Nano

![Platform](https://img.shields.io/badge/Platform-Jetson_Nano-green)
![Python](https://img.shields.io/badge/Python-3.6_|_3.8-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Experimental-orange)
![GUI](https://img.shields.io/badge/Interface-Tkinter-informational)

Classificatore di immagini MRI cerebrali per la rilevazione di tumori, ottimizzato con **TensorRT** su dispositivi **Jetson Nano**.  
Include una GUI locale e supporta modelli personalizzati convertiti da ONNX a motori `.engine`.

---

## 📦 Requisiti
- Jetson Nano con JetPack (consigliato ≥ 4.6.1)
- Python 3.6 o 3.8
- Modello ONNX `brain_mri_resnet18.onnx` presente nella cartella

## 🛠 Installazione dipendenze
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

## ⚙️ Costruzione del motore TensorRT
```bash
python3 build_engine.py
```

Genera `brain_mri_resnet18_v2.engine`.

## 🧠 Avvio classificatore
```bash
python3 classificatore_trt_tumori_v2.py
```

Seleziona un’immagine MRI per ottenere la classificazione ("Tumor" o "No Tumor").

## 📁 Contenuto del pacchetto
- `src/build_engine.py`: costruisce il motore TensorRT da ONNX
- `src/classificatore_trt_tumori_v2.py`: GUI per classificazione con TensorRT
- `src/install_dependencies.sh`: installazione automatica dipendenze
- `notebooks/brain_mri_resnet18_finetuning_v2.ipynb`: notebook Colab per il fine-tuning del modello ResNet-18 su immagini MRI cerebrali. Include data augmentation, addestramento, valutazione ed esportazione in ONNX per l’uso su Jetson Nano.
- `README.md`: istruzioni operative
- `LICENSE`: licenza MIT
- `.gitignore`, `requirements.txt`: per gestione ambiente e progetto

