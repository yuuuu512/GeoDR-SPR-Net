# Predict 208-dimensional enhanced voltages from 40-dimensional inputs.
from pathlib import Path
import numpy as np
import torch
import joblib
from train_geodr_net import DGDenseTransformer

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "data" / "geodr_input.npy"
MODEL_DIR = ROOT / "outputs" / "geodr"
OUTPUT_FILE = ROOT / "outputs" / "enhanced_voltages.npy"


def main():
    x = np.load(INPUT_FILE).astype(np.float32).reshape(-1, 40)
    x_scaler = joblib.load(MODEL_DIR / "scaler_x.pkl")
    y_scaler = joblib.load(MODEL_DIR / "scaler_y.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DGDenseTransformer(H=8, W=5, num_classes=208,
                              local_base_channels=32, local_growth=16,
                              embed_dim=128, num_heads=8, global_layers=2,
                              fusion_layers=1, dropout=0.1,
                              fusion_mode="residual").to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "model.pth", map_location=device, weights_only=True))
    model.eval()
    scaled = x_scaler.transform(x).astype(np.float32).reshape(-1, 1, 8, 5)
    with torch.inference_mode():
        predicted = model(torch.from_numpy(scaled).to(device)).cpu().numpy()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_FILE, y_scaler.inverse_transform(predicted))
    print(f"Saved predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
