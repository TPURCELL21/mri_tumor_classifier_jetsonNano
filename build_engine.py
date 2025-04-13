import tensorrt as trt
import sys
import os


onnx_file = "brain_mri_resnet18.onnx"
engine_file = "brain_mri_resnet18_v2.engine"

if not os.path.exists(onnx_file):
    print(f"❌ File ONNX non trovato: {onnx_file}")
    sys.exit(1)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
print(f"📄 Parsing ONNX model: {onnx_file}")
with open(onnx_file, 'rb') as model:
    if not parser.parse(model.read()):
        print("❌ ERRORE: parsing del file ONNX fallito.")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit(1)

# === Configurazione builder ===
config = builder.create_builder_config()
config.max_workspace_size = 1 << 28  # 256 MiB

# Abilita FP16 se disponibile
if builder.platform_has_fast_fp16:
    print("⚙️ FP16 supportato: abilitato.")
    config.set_flag(trt.BuilderFlag.FP16)
else:
    print("⚠️ FP16 non supportato: uso FP32.")

# === Costruisci il motore ===
print("⚙️ Building TensorRT engine...")
engine = builder.build_engine(network, config)

# === Salva il motore ===
with open(engine_file, "wb") as f:
    f.write(engine.serialize())

print(f"✅ Motore salvato in: {engine_file}")
