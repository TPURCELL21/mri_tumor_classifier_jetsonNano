import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# === Etichette del modello ===
labels = ["No Tumor", "Tumor"]

# === Parametri immagine ===
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (1, 3, 224, 224)
OUTPUT_SHAPE = (1, 2)
TEMPERATURE = 1.5  # <-- valore da ottimizzare offline

# === Preprocessing immagine ===
def preprocess_image(image_path):
    image = Image.open(image_path).resize(IMAGE_SIZE).convert('RGB')
    img = np.asarray(image).astype(np.float32) / 255.0
    img = img.transpose((2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img, dtype=np.float32), image

# === Softmax calibrato ===
def calibrated_softmax(logits, T=TEMPERATURE):
    logits = logits / T
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps, axis=1, keepdims=True)

# === Inizializza motore TensorRT ===
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open("brain_mri_resnet18_v2.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# === Alloca memoria GPU ===
d_input = cuda.mem_alloc(int(np.prod(INPUT_SHAPE)) * 4)
d_output = cuda.mem_alloc(int(np.prod(OUTPUT_SHAPE)) * 4)
bindings = [int(d_input), int(d_output)]

# === GUI ===
class MRIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MRI Tumor Classifier - TensorRT")
        self.label = tk.Label(root, text="Seleziona un'immagine MRI", font=("Helvetica", 14))
        self.label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=300, height=300)
        self.canvas.pack()

        self.result_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.result_label.pack(pady=10)

        self.button = tk.Button(root, text="Seleziona Immagine", command=self.load_image)
        self.button.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Immagini", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        input_data, original_img = preprocess_image(file_path)
        img = original_img.resize((300, 300))
        self.canvas.delete("all")
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(150, 150, image=self.tk_image)

        # Inferenza
        output_data = np.empty(OUTPUT_SHAPE, dtype=np.float32)
        cuda.memcpy_htod(d_input, input_data)
        try:
            context.execute_v2(bindings)
        except Exception as e:
            self.result_label.config(text=f"Errore: {e}", fg="orange")
            return
        cuda.memcpy_dtoh(output_data, d_output)

        # Calibrazione delle probabilità
        probs = calibrated_softmax(output_data)
        pred_class = int(np.argmax(probs))
        confidence = float(probs[0][pred_class])
        label = labels[pred_class]
        self.result_label.config(
            text=f"{label} ({confidence * 100:.2f}%)",
            fg="green" if pred_class == 0 else "red"
        )

# === Avvio interfaccia ===
if __name__ == "__main__":
    root = tk.Tk()
    app = MRIApp(root)
    root.mainloop()

