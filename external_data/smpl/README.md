# SMPL-H body model

This folder is **not** stored in git (licensed assets, ~500 MB).

## Download (manual — license required)

1. Register at [MANO](https://mano.is.tue.mpg.de/download.php).
2. Download:
   - **Extended SMPL+H model** (body without hands)
   - **Models & Code** (hand models)
3. Extract into this directory so you have:

```
external_data/smpl/smplh/neutral/model.npz
external_data/smpl/mano_v1_2/models/MANO_LEFT.pkl
external_data/smpl/mano_v1_2/models/MANO_RIGHT.pkl
```

## Build the combined model

After the source files above are in place:

```bash
python scripts/setup_data.py --build-smplh
```

This creates `external_data/smpl/SMPLH_NEUTRAL.pkl`, which the retargeting scripts use.

See also: [loco-mujoco SMPL setup guide](../../loco-mujoco/loco_mujoco/smpl/README.MD).
