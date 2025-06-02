import torch.nn.utils.prune as prune
import torch.quantization
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

# Use same directory as MODEL_PATH for exports
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMPORT_DIR = "/Users/joshfeather/Documents/ball_tracking_protoyping/trajectory_models/tennis_trajectory_model_bi_lstm.pth"
EXPORT_DIR = "/Users/joshfeather/Documents/ball_tracking_protoyping/exported_models"
os.makedirs(EXPORT_DIR, exist_ok=True)
BASE_NAME = os.path.splitext(os.path.basename(IMPORT_DIR))[0]

# === Model Definition ===
class BiLSTM_Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.fc(out)

# === Optimization Functions ===
def prune_model(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.LSTM):
            prune.l1_unstructured(module, name="weight_hh_l0", amount=0.2)
            prune.remove(module, "weight_hh_l0")
    print("✅ Pruning applied.")
    return model

def quantise_model(model):
    model.cpu()
    model.eval()
    torch.backends.quantized.engine = 'qnnpack'  # Fix for quant engine error
    quant_model = torch.quantization.quantize_dynamic(
        model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
    )
    print("✅ Quantisation applied.")
    return quant_model

def export_onnx_model(model, onnx_path):
    model.eval()
    dummy_input = torch.randn(1, 240, 3)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {1: "seq_len"}, "output": {1: "seq_len"}},
        opset_version=11
    )
    print(f"✅ ONNX export complete: {onnx_path}")

# === Pipeline Start ===
print("\n🚀 Optimising model for deployment...")

# 1. Load original model
original_model = BiLSTM_Model()
original_model.load_state_dict(torch.load(IMPORT_DIR, map_location=device))

# 2. Apply pruning
pruned_model = prune_model(original_model)
pruned_path = os.path.join(EXPORT_DIR, "pruned.pth")
torch.save(pruned_model.state_dict(), pruned_path)
print(f"💾 Pruned model saved: {pruned_path}")

# 3. Export ONNX from the pruned (not quantised) model
onnx_path = os.path.join(EXPORT_DIR, "pruned.onnx")
export_onnx_model(pruned_model, onnx_path)

# 4. Apply quantisation (for PyTorch inference)
quant_model = quantise_model(pruned_model)
quant_path = os.path.join(EXPORT_DIR, "quantised.pth")
torch.save(quant_model.state_dict(), quant_path)
print(f"💾 Quantised model saved: {quant_path}")

print("\n✅ All optimised versions saved in:", EXPORT_DIR)
