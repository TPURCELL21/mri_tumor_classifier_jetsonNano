# MRI Tumor Classifier - TensorRT su Jetson Nano

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
- `build_engine.py`: costruisce il motore TensorRT da ONNX
- `classificatore_trt_tumori_v2.py`: GUI per classificazione con TensorRT
- `install_dependencies.sh`: installazione automatica dipendenze
- `README.md`: istruzioni operative

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

