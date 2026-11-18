# GeoDR-Net: sparse 8-electrode voltage (40) -> enhanced 16-electrode voltage (208).
# Core architecture extracted from the research implementation.
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

def ensure_divisible(x, divisor):
    return int(math.ceil(x / divisor) * divisor)


class _DenseLayer(nn.Module):

    def __init__(self, in_channels, growth_rate=16, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.act2 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, 3, padding=1, bias=False)
        self.drop = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))
        out = self.drop(out)
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):

    def __init__(self, num_layers, in_channels, growth_rate=16, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            layer = _DenseLayer(channels, growth_rate, bn_size, drop_rate)
            self.layers.append(layer)
            channels += growth_rate
        self.out_channels = channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):

    def __init__(self, in_channels, out_channels=None, compression=0.5):
        super().__init__()
        if out_channels is None:
            out_channels = int(in_channels * compression)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ELU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(self.act(self.norm(x)))
        x = self.pool(x)
        return x


class GeometryPositionalEmbedding(nn.Module):

    def __init__(self, H, W, embed_dim, hidden=64):
        super().__init__()
        self.H = H
        self.W = W
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(nn.Linear(4, hidden), nn.ELU(), nn.Linear(hidden, embed_dim))
        coords = self._build_coords(H, W)
        self.register_buffer('coords', coords)

    def _build_coords(self, H, W):
        ys = torch.linspace(-1.0, 1.0, H)
        xs = torch.linspace(-1.0, 1.0, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        x = grid_x.reshape(-1)
        y = grid_y.reshape(-1)
        r = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x) / math.pi
        coords = torch.stack([x, y, r, theta], dim=1)
        return coords.float()

    def forward(self, batch_size, device=None):
        coords = self.coords.to(device) if device is not None else self.coords
        emb = self.mlp(coords)
        emb = emb.unsqueeze(0).expand(batch_size, -1, -1)
        return emb


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads=4, ff_hidden=None, dropout=0.1):
        super().__init__()
        if ff_hidden is None:
            ff_hidden = dim * 2
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_hidden), nn.ELU(), nn.Linear(ff_hidden, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=4, ff_hidden=None, dropout=0.1):
        super().__init__()
        if ff_hidden is None:
            ff_hidden = dim * 2
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_hidden), nn.ELU(), nn.Linear(ff_hidden, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, kv):
        attn_out, _ = self.attn(query, kv, kv)
        x = self.norm1(query + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class LocalConvEncoder(nn.Module):

    def __init__(self, in_ch=1, base_channels=32, growth_rate=16, drop_rate=0.0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, base_channels, 3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ELU(inplace=True))
        self.db1 = _DenseBlock(2, base_channels, growth_rate)
        self.tr1 = _Transition(self.db1.out_channels, out_channels=max(8, self.db1.out_channels // 2))
        self.db2 = _DenseBlock(2, self.tr1.out_channels, growth_rate)
        self.out_channels = self.db2.out_channels

    def forward(self, x):
        x = self.stem(x)
        x = self.db1(x)
        x = self.tr1(x)
        x = self.db2(x)
        return x


class GlobalTransformerEncoder(nn.Module):

    def __init__(self, embed_dim, num_layers=2, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.out_dim = embed_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DGDenseTransformer(nn.Module):

    def __init__(self, H=8, W=5, num_classes=208, local_base_channels=32, local_growth=16, embed_dim=128, num_heads=8, global_layers=2, fusion_layers=1, dropout=0.1, fusion_mode='alpha'):
        super().__init__()
        if embed_dim % num_heads != 0:
            embed_dim = ensure_divisible(embed_dim, num_heads)
        self.H = H
        self.W = W
        self.embed_dim = embed_dim
        self.fusion_mode = fusion_mode
        self.local_encoder = LocalConvEncoder(1, local_base_channels, local_growth)
        self.local_proj = nn.Linear(self.local_encoder.out_channels, embed_dim)
        self.global_stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ELU(inplace=True), nn.AvgPool2d(2))
        self.global_proj_conv = nn.Conv2d(16, embed_dim, 1, bias=False)
        self.geom_pos = GeometryPositionalEmbedding(H, W, embed_dim)
        self.global_encoder = GlobalTransformerEncoder(embed_dim, num_layers=global_layers, num_heads=num_heads)
        self.cross_glob_to_local = CrossAttentionBlock(embed_dim, num_heads=num_heads)
        self.cross_local_to_glob = CrossAttentionBlock(embed_dim, num_heads=num_heads)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.fusion_mlp = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ELU(), nn.Linear(embed_dim, embed_dim))
        self.post_transformer = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(fusion_layers)])
        self.head = nn.Sequential(nn.LayerNorm(embed_dim * H * W), nn.Linear(embed_dim * H * W, num_classes))

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        local_feats = self.local_encoder(x)
        local_feats = F.interpolate(local_feats, size=(self.H, self.W), mode='bilinear', align_corners=False)
        C_local = local_feats.shape[1]
        local_seq = local_feats.view(B, C_local, -1).permute(0, 2, 1)
        local_emb = self.local_proj(local_seq)
        g = self.global_stem(x)
        g = self.global_proj_conv(g)
        g = F.interpolate(g, size=(self.H, self.W), mode='bilinear', align_corners=False)
        global_seq = g.view(B, self.embed_dim, -1).permute(0, 2, 1)
        pos = self.geom_pos(B, device=device)
        local_emb += pos
        global_seq += pos
        global_encoded = self.global_encoder(global_seq)
        glob_attended = self.cross_glob_to_local(global_encoded, local_emb)
        local_attended = self.cross_local_to_glob(local_emb, global_encoded)
        if self.fusion_mode == 'alpha':
            fused = self.alpha * glob_attended + (1 - self.alpha) * local_attended
        elif self.fusion_mode == 'residual':
            fused = self.alpha * (glob_attended + global_encoded) + (1 - self.alpha) * (local_attended + local_emb)
        else:
            fused = torch.cat([glob_attended, local_attended], dim=-1)
            fused = self.fusion_mlp(fused)
        for block in self.post_transformer:
            fused = block(fused)
        x_flat = fused.view(B, -1)
        out = self.head(x_flat)
        return out

# ---------------------- Minimal training example ----------------------
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "geodr_train.npz"
OUTPUT_DIR = ROOT / "outputs" / "geodr"
SEED = 116
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 1e-4


def main():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Training data not found: {DATA_FILE}. See README.md.")

    with np.load(DATA_FILE) as ds:
        x = np.asarray(ds["X"], dtype=np.float32).reshape(-1, 40)
        y = np.asarray(ds["Y"], dtype=np.float32).reshape(-1, 208)
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
    x_norm = x_scaler.transform(x).astype(np.float32).reshape(-1, 1, 8, 5)
    y_norm = y_scaler.transform(y).astype(np.float32)
    tensors = TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(y_norm))
    train_loader = DataLoader(torch.utils.data.Subset(tensors, train_ids.tolist()), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(tensors, val_ids.tolist()), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DGDenseTransformer(H=8, W=5, num_classes=208, local_base_channels=32, local_growth=16, embed_dim=128, num_heads=8, global_layers=2, fusion_layers=1, dropout=0.1, fusion_mode="residual").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    loss_fn = nn.SmoothL1Loss()
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
