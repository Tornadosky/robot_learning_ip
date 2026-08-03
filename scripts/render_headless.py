"""Headless render wrapper.

This OpenCV build has no GUI backend, but loco-mujoco's VideoRecorder.stop()
calls cv2.destroyAllWindows() (a GUI call) before releasing the writer, which
crashes headless and aborts the video. Stub the GUI no-ops, then run the normal
render entrypoint with the same CLI args. Use with MUJOCO_GL=osmesa.

    MUJOCO_GL=osmesa python render_headless.py --mode grid --robot g1 ...
"""
from __future__ import annotations

import runpy
from pathlib import Path

import cv2

for _fn in ("destroyAllWindows", "destroyWindow", "imshow", "namedWindow",
            "waitKey", "startWindowThread", "setWindowProperty"):
    setattr(cv2, _fn, lambda *a, **k: 0)

_target = str(Path(__file__).resolve().parent / "render_morphology_deepmimic.py")
runpy.run_path(_target, run_name="__main__")
