# SPR-Net: 16-electrode voltage (208) -> conductivity field (2334).
# Core architecture extracted from the research implementation.
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

class _DenseLayer(nn.Module):

    def __init__(self, in_channels, growth_rate=16, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.act2 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(p=drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):

    def __init__(self, num_layers, in_channels, growth_rate=16, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            layer = _DenseLayer(channels, growth_rate, bn_size, drop_rate)
            layers.append(layer)
            channels += growth_rate
        self.layers = nn.ModuleList(layers)
        self.out_channels = channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):

    def __init__(self, in_channels, compression=0.5):
        super().__init__()
        out_channels = int(in_channels * compression)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ELU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(self.act(self.norm(x)))
        x = self.pool(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads=4, ff_hidden=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_hidden), nn.ELU(), nn.Linear(ff_hidden, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DenseTransformerNet16x13(nn.Module):

    def __init__(self, num_classes=2334, growth_rate=16, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.ELU(inplace=True))
        self.db1 = _DenseBlock(num_layers=4, in_channels=32, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        c1 = self.db1.out_channels
        self.tr1 = _Transition(c1, compression=0.5)
        c1_tr = self.tr1.out_channels
        self.db2 = _DenseBlock(num_layers=6, in_channels=c1_tr, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        c2 = self.db2.out_channels
        self.tr2 = _Transition(c2, compression=0.5)
        c2_tr = self.tr2.out_channels
        self.db3 = _DenseBlock(num_layers=8, in_channels=c2_tr, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        c3 = self.db3.out_channels
        self.final_channels = c3
        self.trans_dim = self.final_channels
        self.trans_block = TransformerBlock(dim=self.trans_dim, num_heads=4, ff_hidden=512, dropout=0.1)
        self._init_fc_layer(num_classes)
        self.out_act = nn.Sigmoid()

    def _forward_features(self, x):
        x = self.stem(x)
        x = self.db1(x)
        x = self.tr1(x)
        x = self.db2(x)
        x = self.tr2(x)
        x = self.db3(x)
        return x

    def _init_fc_layer(self, num_classes):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 16, 13)
            feats = self._forward_features(dummy)
            B, C, H, W = feats.shape
            self.trans_seq_len = H * W
        self.pos_embed = nn.Parameter(torch.zeros(1, self.trans_seq_len, C))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.fc = nn.Linear(C * self.trans_seq_len, num_classes)

    def forward(self, x):
        x = self._forward_features(x)
        B, C, H, W = x.shape
        seq_len = H * W
        x = x.view(B, C, seq_len).permute(0, 2, 1)
        if seq_len != self.pos_embed.shape[1]:
            pos = self.pos_embed
            orig_h = orig_w = int(self.pos_embed.shape[1] ** 0.5)
            pos = pos.permute(0, 2, 1).view(1, C, orig_h, orig_w)
            pos = F.interpolate(pos, size=(H, W), mode='bilinear', align_corners=False)
            pos = pos.view(1, C, seq_len).permute(0, 2, 1)
            x = x + pos
        else:
            x = x + self.pos_embed
        x = self.trans_block(x)
        x = x.permute(0, 2, 1).contiguous().view(B, -1)
        x = self.out_act(self.fc(x))
        return x

# ---------------------- Minimal training example ----------------------
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "spr_train.npz"
OUTPUT_DIR = ROOT / "outputs" / "spr"
SEED = 116
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 1e-4


def main():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Training data not found: {DATA_FILE}. See README.md.")

    with np.load(DATA_FILE) as ds:
        x = np.asarray(ds["X"], dtype=np.float32).reshape(-1, 208)
        y = np.asarray(ds["Y"], dtype=np.float32).reshape(-1, 2334)
    if len(x) != len(y) or len(x) < 10:
        raise ValueError("X and Y must contain the same number of samples (at least 10).")

    # Reproducible sample-level 80/10/10 train/validation/test partition.
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    order = np.random.RandomState(SEED).permutation(len(x))
    n_train = int(0.8 * len(x))
    n_val = int(0.1 * len(x))
    train_ids = order[:n_train]
    val_ids = order[n_train:n_train + n_val]

    x_scaler = MinMaxScaler(feature_range=(0, 1)).fit(x[train_ids])
    y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(y[train_ids])
    x_norm = x_scaler.transform(x).astype(np.float32).reshape(-1, 1, 16, 13)
    y_norm = y_scaler.transform(y).astype(np.float32)
    tensors = TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(y_norm))
    train_loader = DataLoader(torch.utils.data.Subset(tensors, train_ids.tolist()), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(tensors, val_ids.tolist()), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseTransformerNet16x13(num_classes=2334, growth_rate=16, bn_size=4, drop_rate=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    loss_fn = nn.MSELoss()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(x_scaler, OUTPUT_DIR / "scaler_x.pkl")
    joblib.dump(y_scaler, OUTPUT_DIR / "scaler_y.pkl")

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(xb)

        model.eval()
        val_total = 0.0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_total += loss_fn(model(xb), yb).item() * len(xb)
        train_loss = train_total / len(train_ids)
        val_loss = val_total / len(val_ids)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "model.pth")
        print(f"Epoch {epoch:03d}/{EPOCHS}: train={train_loss:.6g} val={val_loss:.6g}")

    print(f"Saved model and scalers to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
