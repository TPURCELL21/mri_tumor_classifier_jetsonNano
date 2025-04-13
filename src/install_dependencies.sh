#!/bin/bash
echo "⚙️ Installazione dipendenze per MRI Classifier GUI..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Completato! Ricorda di installare manualmente TensorRT e PyCUDA se necessario."

