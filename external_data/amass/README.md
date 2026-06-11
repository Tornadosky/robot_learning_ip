# AMASS motion capture data

This folder is **not** stored in git (~2+ GB for the DanceDB subset used here).

## Download (manual — license required)

1. Register at [AMASS](https://amass.is.tue.mpg.de/index.html).
2. Download the **DanceDB** dataset in **SMPL-H G** format.
3. Extract so clips live under this directory, e.g.:

```
external_data/amass/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses.npz
```

## Minimum clip for the bundled scripts

The default AMASS retarget scripts expect:

```
DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses
```

Downloading the full DanceDB subset is recommended so other clips work without extra setup.

## Verify

After placing files here, run:

```bash
python scripts/setup_data.py --check-only
```
