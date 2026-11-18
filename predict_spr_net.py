# Reconstruct 2334-dimensional conductivity fields from 208-dimensional voltages.
from pathlib import Path
import numpy as np
import torch
import joblib
from train_spr_net import DenseTransformerNet16x13

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "data" / "spr_input.npy"
MODEL_DIR = ROOT / "outputs" / "spr"
OUTPUT_FILE = ROOT / "outputs" / "reconstructed_conductivity.npy"


def main():
    x = np.load(INPUT_FILE).astype(np.float32).reshape(-1, 208)
    x_scaler = joblib.load(MODEL_DIR / "scaler_x.pkl")
    y_scaler = joblib.load(MODEL_DIR / "scaler_y.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseTransformerNet16x13(num_classes=2334, growth_rate=16,
                                     bn_size=4, drop_rate=0.0).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "model.pth", map_location=device, weights_only=True))
    model.eval()
    scaled = x_scaler.transform(x).astype(np.float32).reshape(-1, 1, 16, 13)
    with torch.inference_mode():
        predicted = model(torch.from_numpy(scaled).to(device)).cpu().numpy()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_FILE, y_scaler.inverse_transform(predicted))
    print(f"Saved predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
