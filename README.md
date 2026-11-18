# GeoDR-Net and SPR-Net — source code

This compact package contains the core network architectures and simple training and inference examples for the two-stage EIT framework.

- GeoDR-Net: 8-electrode input (8 × 5 = 40 values) → 16-electrode voltage representation (16 × 13 = 208 values).
- SPR-Net: 16-electrode voltage representation (208 values) → conductivity distribution (2334 values).

## Files

- `train_geodr_net.py`: GeoDR-Net architecture and example training.
- `predict_geodr_net.py`: GeoDR-Net inference.
- `train_spr_net.py`: SPR-Net architecture and example training.
- `predict_spr_net.py`: SPR-Net inference.

## Installation

`pip install -r requirements.txt`

## Input data

Place files in `data/` (the repository does not include measured or simulation data):

- `geodr_train.npz`: arrays `X` of shape `(N, 40)` and `Y` of shape `(N, 208)`.
- `spr_train.npz`: arrays `X` of shape `(N, 208)` and `Y` of shape `(N, 2334)`.
- `geodr_input.npy`: 40-element vectors, shape `(N, 40)` or `(40,)`.
- `spr_input.npy`: 208-element vectors, shape `(N, 208)` or `(208,)`.

The voltage channel order must follow the excitation–measurement protocol specified in the manuscript. The 2334 output components must correspond to the mesh-node ordering used for training.

## Usage

```bash
python train_geodr_net.py
python predict_geodr_net.py
python train_spr_net.py
python predict_spr_net.py
```

The training examples use AdamW, 200 epochs, batch size 16, seed 116, an 80/10/10 sample-level partition, and training-subset-only MinMax scaling. Model weights and scalers are written to `outputs/geodr/` and `outputs/spr/`.

Prediction outputs are saved as `.npy` arrays in `outputs/`. You can supply `outputs/enhanced_voltages.npy` as `data/spr_input.npy` to run the full two-stage inference after training the models. For separately trained networks, confirm that the correct input scaler is used at each stage.

This is a streamlined reference implementation focusing on the core model architectures, not an archive of every experiment or dataset. Original private paths, equipment-control code and third-party measurement data are not distributed. It does not claim to reproduce numerical tables without the corresponding experimental data and training setup.
